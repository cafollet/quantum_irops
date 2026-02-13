"""post_analysis.py

Text-based post-analysis of pipeline re-accommodation results against the
Copa Airlines Quantum Solution – Passenger Re-accommodation Rules Set (Phase 2).

Designed for direct integration with the Shiny frontend dashboard.
All output is returned as a formatted string (no printing, no plotting).

Public API
----------
    run_post_analysis(assignments_df, unbooked_df, available_flights)
        -> str   (the full report)

`available_flights` may be a file path (str/Path) or an already-loaded
pandas or polars DataFrame.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rule-Set 1 constants
# ---------------------------------------------------------------------------
DELAY_BANDS = [(6, 70), (12, 50), (24, 40), (48, 30), (72, 30)]
MAX_DELAY_HRS = 72
CITY_PAIR_SAME = 40
CITY_PAIR_DIFF = 20
STOPOVER_PENALTY = -20
GRADE_CUTS = [("A", 150), ("B", 130), ("C", 100)]

DATETIME_FMTS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%Y-%m-%d",
]

# CVM priority tiers — boundaries chosen to reflect Copa's CVM scale (0-~13)
# Higher CVM = higher-value passenger = should be reassigned first.
CVM_TIERS = [
    ("Low",     0,   3),
    ("Medium",  3,   6),
    ("High",    6,   9),
    ("Premium", 9,   float("inf")),
]

ASGN_REQUIRED = [
    "ALT_DEP_KEY", "ALT_CABIN_CD", "PAX_CNT",
    "ORIG_CD", "DEST_CD", "ALT_ORIG_CD", "ALT_DEST_CD",
    "ARR_DTML", "ALT_ARR_DTML", "DEP_DTML", "ALT_DEP_DTML",
    "IS_DIRECT", "CABIN_CD",
]
AVAIL_REQUIRED = [
    "DEP_KEY",
    "C_CAP_CNT", "C_AUL_CNT", "C_PAX_CNT", "C_AVAIL_CNT",
    "Y_CAP_CNT", "Y_AUL_CNT", "Y_PAX_CNT", "Y_AVAIL_CNT",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_pandas(df) -> pd.DataFrame:
    """Accept pandas or polars DataFrame; always return pandas."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df.copy()


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_dt(series: pd.Series) -> pd.Series:
    for fmt in DATETIME_FMTS:
        try:
            r = pd.to_datetime(series, format=fmt, errors="coerce")
            if r.notna().mean() > 0.5:
                return r
        except Exception:
            pass
    return pd.to_datetime(series, errors="coerce")


def _delay_score(h: float) -> int:
    """Score based on delay hours.  h <= 0 (early/on-time) -> best score."""
    if h <= 0:
        return DELAY_BANDS[0][1]
    if h > MAX_DELAY_HRS:
        return 0
    for thr, pts in DELAY_BANDS:
        if h <= thr:
            return pts
    return 30


def _grade(s: float) -> str:
    for ltr, thr in GRADE_CUTS:
        if s >= thr:
            return ltr
    return "D"


def _short_key(key, n: int = 14) -> str:
    s = str(key)
    if len(s) >= 10 and s[2:10].isdigit():
        s = s[:2] + s[10:]
    return s[-n:] if len(s) > n else s


# ---------------------------------------------------------------------------
# 1. Overbooking check  (most critical)
# ---------------------------------------------------------------------------

def _check_overbooking(df: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["ALT_DEP_KEY", "ALT_CABIN_CD"]).copy()
    work = work[work["ALT_CABIN_CD"].astype(str).str.strip() != ""]
    if work.empty:
        raise ValueError("No valid (ALT_DEP_KEY, ALT_CABIN_CD) pairs in assignments.")

    new_load = (
        work.groupby(["ALT_DEP_KEY", "ALT_CABIN_CD"])["PAX_CNT"]
        .sum()
        .reset_index()
        .rename(columns={"PAX_CNT": "new_pax"})
    )

    merged = new_load.merge(avail, left_on="ALT_DEP_KEY", right_on="DEP_KEY", how="left")

    records = []
    for _, r in merged.iterrows():
        cab = str(r["ALT_CABIN_CD"]).upper()
        if cab == "C":
            baseline, aul, cap, avail_seats = (
                r["C_PAX_CNT"], r["C_AUL_CNT"], r["C_CAP_CNT"], r["C_AVAIL_CNT"]
            )
        else:
            baseline, aul, cap, avail_seats = (
                r["Y_PAX_CNT"], r["Y_AUL_CNT"], r["Y_CAP_CNT"], r["Y_AVAIL_CNT"]
            )
        total_after = baseline + r["new_pax"]
        ob = max(0, total_after - aul)
        records.append(
            {
                "ALT_DEP_KEY": r["ALT_DEP_KEY"],
                "ALT_CABIN_CD": cab,
                "new_pax": int(r["new_pax"]),
                "baseline_pax": int(baseline),
                "aul": int(aul),
                "phys_cap": int(cap),
                "avail_at_solve": int(avail_seats),
                "total_after": int(total_after),
                "overbooked_by": int(ob),
                "util_pct": round(100 * total_after / aul, 1) if aul else None,
                "is_overbooked": ob > 0,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("overbooked_by", ascending=False)
        .reset_index(drop=True)
    )


def _annotate_overbooking(df: pd.DataFrame, ob: pd.DataFrame) -> pd.DataFrame:
    lookup = ob.set_index(["ALT_DEP_KEY", "ALT_CABIN_CD"])[
        ["overbooked_by", "is_overbooked", "util_pct"]
    ]
    df = df.copy().join(lookup, on=["ALT_DEP_KEY", "ALT_CABIN_CD"], how="left")
    df["overbooked_by"] = df["overbooked_by"].fillna(0)
    df["is_overbooked"] = df["is_overbooked"].fillna(False)
    return df


# ---------------------------------------------------------------------------
# 2. Rule Set 1 - flight quality scoring
# ---------------------------------------------------------------------------

def _score_flight_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["arr_delay_hrs"] = np.nan
    m = df["ARR_DTML"].notna() & df["ALT_ARR_DTML"].notna()
    df.loc[m, "arr_delay_hrs"] = (
        (df.loc[m, "ALT_ARR_DTML"] - df.loc[m, "ARR_DTML"]).dt.total_seconds() / 3600
    )

    df["dep_delay_hrs"] = np.nan
    m2 = df["DEP_DTML"].notna() & df["ALT_DEP_DTML"].notna()
    df.loc[m2, "dep_delay_hrs"] = (
        (df.loc[m2, "ALT_DEP_DTML"] - df.loc[m2, "DEP_DTML"]).dt.total_seconds() / 3600
    )

    df["eligible"] = df["arr_delay_hrs"].apply(
        lambda h: h <= MAX_DELAY_HRS if pd.notna(h) else True
    )
    df["score_arr_delay"] = df["arr_delay_hrs"].apply(
        lambda h: _delay_score(h) if pd.notna(h) else 0
    )
    df["score_dep_delay"] = df["dep_delay_hrs"].apply(
        lambda h: _delay_score(h) if pd.notna(h) else 0
    )

    orig_pair = df["ORIG_CD"].astype(str) + "-" + df["DEST_CD"].astype(str)
    alt_pair = df["ALT_ORIG_CD"].astype(str) + "-" + df["ALT_DEST_CD"].astype(str)
    df["score_city_pair"] = np.where(orig_pair == alt_pair, CITY_PAIR_SAME, CITY_PAIR_DIFF)
    df["city_pair_match"] = orig_pair == alt_pair
    df["score_stopover"] = df["IS_DIRECT"].apply(lambda d: 0 if d else STOPOVER_PENALTY)

    df["quality_score"] = (
        df["score_arr_delay"]
        + df["score_dep_delay"]
        + df["score_city_pair"]
        + df["score_stopover"]
    )
    df["quality_grade"] = df["quality_score"].apply(_grade)
    return df


# ---------------------------------------------------------------------------
# 3. Rule Set 3 - cabin compliance
# ---------------------------------------------------------------------------

def _score_cabin_compliance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ORIG_CABIN" in df.columns:
        orig = df["ORIG_CABIN"].fillna(df["CABIN_CD"]).astype(str).str.upper()
    else:
        orig = df["CABIN_CD"].astype(str).str.upper()
    alt = df["ALT_CABIN_CD"].astype(str).str.upper()
    result = []
    for o, a in zip(orig, alt):
        if o == "Y":
            result.append("same" if a == "Y" else "upgrade_violation")
        elif o in ("C", "F"):
            result.append("downgrade" if a == "Y" else "same")
        else:
            result.append("same")
    df["cabin_change"] = result
    df["cabin_ok"] = df["cabin_change"] != "upgrade_violation"
    return df


# ---------------------------------------------------------------------------
# 4. Journey scenario (Rule Set 4)
# ---------------------------------------------------------------------------

def _classify_journey(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    orig_direct = (
        df.get("PREV_OD_BROKEN_IND", pd.Series(0, index=df.index))
        .fillna(0)
        .astype(int)
        == 0
    )
    alt_direct = df["IS_DIRECT"].astype(bool)

    def _s(od, ad):
        if od and ad:
            return "1-1 (One-One)"
        if od and not ad:
            return "1-Multi (One-Multi)"
        if not od and ad:
            return "Multi-1 (Multi-One)"
        return "Multi-Multi"

    df["journey_scenario"] = [_s(od, ad) for od, ad in zip(orig_direct, alt_direct)]
    return df


# ---------------------------------------------------------------------------
# CVM tier breakdown (PAX_CNT-weighted so multi-pax PNRs count correctly)
# ---------------------------------------------------------------------------

def _cvm_tier_rows(df_assigned, df_unbooked):
    """Return one dict per CVM tier with reassignment counts and rate."""
    rows = []
    for name, lo, hi in CVM_TIERS:
        a_mask = (df_assigned["CVM"] >= lo) & (df_assigned["CVM"] < hi)
        a_pax = int(df_assigned.loc[a_mask, "PAX_CNT"].sum())
        if df_unbooked is not None and len(df_unbooked):
            u_mask = (df_unbooked["CVM"] >= lo) & (df_unbooked["CVM"] < hi)
            u_pax = int(df_unbooked.loc[u_mask, "PAX_CNT"].sum())
        else:
            u_pax = 0
        total = a_pax + u_pax
        rows.append(
            {
                "tier": name,
                "lo": lo,
                "hi": hi,
                "assigned": a_pax,
                "unbooked": u_pax,
                "total": total,
                "rate": 100.0 * a_pax / total if total else 0.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _compute_metrics(df, ob, unbooked):
    n_asgn = int(df["PAX_CNT"].sum()) if "PAX_CNT" in df.columns else len(df)
    n_unb = (
        int(unbooked["PAX_CNT"].sum())
        if unbooked is not None and len(unbooked)
        else 0
    )
    total = n_asgn + n_unb

    w = (
        df["PAX_CNT"] * df["CVM"]
        if "CVM" in df.columns and "PAX_CNT" in df.columns
        else None
    )
    ws = w.sum() if w is not None and w.sum() > 0 else None

    sc = df["journey_scenario"].value_counts().to_dict()
    preferred = sc.get("1-1 (One-One)", 0) + sc.get("Multi-1 (Multi-One)", 0)

    n_ob = int(ob["is_overbooked"].sum())
    seats_ob = int(ob["overbooked_by"].sum())

    # PAX-weighted mean CVM: headline check that priority ordering is working
    wtd_cvm_asgn = float(
        (df["CVM"] * df["PAX_CNT"]).sum() / df["PAX_CNT"].sum()
    ) if "CVM" in df.columns and df["PAX_CNT"].sum() > 0 else np.nan
    wtd_cvm_unb = float(
        (unbooked["CVM"] * unbooked["PAX_CNT"]).sum() / unbooked["PAX_CNT"].sum()
    ) if (
        unbooked is not None and len(unbooked)
        and "CVM" in unbooked.columns and unbooked["PAX_CNT"].sum() > 0
    ) else np.nan

    return {
        "n_assigned": n_asgn,
        "n_unbooked": n_unb,
        "total": total,
        "reaccom_pct": 100 * n_asgn / total if total else 0,
        "mean_arr_delay": df["arr_delay_hrs"].mean(),
        "median_arr_delay": df["arr_delay_hrs"].median(),
        "wtd_arr_delay": (df["arr_delay_hrs"] * w).sum() / ws if ws else np.nan,
        "mean_quality": df["quality_score"].mean(),
        "wtd_quality": (df["quality_score"] * w).sum() / ws if ws else np.nan,
        "grades": df["quality_grade"].value_counts().to_dict(),
        "scenarios": sc,
        "preferred_pct": 100 * preferred / len(df) if len(df) else 0,
        "cabin_ok_pct": 100 * df["cabin_ok"].mean(),
        "ineligible": int((~df["eligible"]).sum()),
        "n_ob_flights": n_ob,
        "seats_overbooked": seats_ob,
        "pct_ob_flights": 100 * n_ob / len(ob) if len(ob) else 0,
        "max_util": float(ob["util_pct"].max()) if len(ob) else 0,
        "n_flights_total": len(ob),
        "wtd_cvm_assigned": wtd_cvm_asgn,
        "wtd_cvm_unbooked": wtd_cvm_unb,
    }


# ---------------------------------------------------------------------------
# Text report builder
# ---------------------------------------------------------------------------

def _format_report(df, ob, m, unb=None):
    buf = io.StringIO()
    w = buf.write
    SEP = "=" * 72
    THIN = "-" * 72

    w(f"\n{SEP}\n")
    w("  RE-ACCOMMODATION QUALITY REPORT  (Phase 2 Rules Set)\n")
    w(f"{SEP}\n")

    # --- Headline: passenger reassignment rate (individual people, not PNRs) ---
    w(f"\n  {'Passengers reassigned':<30}: {m['n_assigned']:,} / {m['total']:,}")
    w(f"  ({m['reaccom_pct']:.1f}%)\n")
    w(f"  {'Passengers still unbooked':<30}: {m['n_unbooked']:,}\n")

    # --- Overbooking summary (counts only, no per-flight table) ---
    w(f"\n  {'Flight-cabin pairs checked':<30}: {m['n_flights_total']}\n")
    w(f"  {'Overbooked pairs':<30}: {m['n_ob_flights']}\n")
    w(f"  {'Total seats over AUL':<30}: {m['seats_overbooked']}\n")
    w(f"  {'Peak utilisation':<30}: {m['max_util']:.0f}%\n")
    if m["n_ob_flights"] > 0:
        w(
            f"\n  *** OVERBOOKING DETECTED — {m['n_ob_flights']} of"
            f" {m['n_flights_total']} pairs exceed authorised load ***\n"
        )
    else:
        w(f"\n  OK  All {m['n_flights_total']} flight-cabin pairs within authorised load\n")

    # --- CVM priority breakdown ---
    w(f"\n  {THIN}\n")
    w("  REASSIGNMENT BY CVM TIER  (higher CVM = higher priority)\n")
    w(f"  {THIN}\n")

    # Headline delta: are higher-value passengers actually getting seats first?
    if not np.isnan(m["wtd_cvm_assigned"]):
        w(f"  Mean CVM assigned : {m['wtd_cvm_assigned']:.2f}")
        if not np.isnan(m["wtd_cvm_unbooked"]):
            delta = m["wtd_cvm_assigned"] - m["wtd_cvm_unbooked"]
            sign = "+" if delta >= 0 else ""
            verdict = (
                "priority ordering OK"
                if delta > 0
                else "WARNING: lower-value pax favoured"
            )
            w(
                f"  |  unbooked : {m['wtd_cvm_unbooked']:.2f}"
                f"  |  delta {sign}{delta:.2f}  ({verdict})\n"
            )
        else:
            w("\n")

    w(
        f"\n  {'Tier':<10} {'CVM range':<12} {'Assigned':>10}"
        f" {'Unbooked':>10} {'Total':>8} {'Rate':>7}  Bar\n"
    )
    w("  " + "-" * 64 + "\n")
    for t in _cvm_tier_rows(df, unb):
        hi_str = f"{int(t['hi'])}" if t["hi"] != float("inf") else "+"
        rng = f"{int(t['lo'])}-{hi_str}"
        bar = "#" * int(t["rate"] / 5)  # one # per 5 %
        w(
            f"  {t['tier']:<10} {rng:<12} {t['assigned']:>10,}"
            f" {t['unbooked']:>10,} {t['total']:>8,} {t['rate']:>6.1f}%  {bar}\n"
        )

    # --- Rule Set 4: solution ranking ---
    w(f"\n  {THIN}\n")
    w("  SOLUTION RANKING  (Rule Set 4)\n")
    w(f"  {THIN}\n")
    w(f"  Mean arrival delay    : {m['mean_arr_delay']:.2f} h  (median {m['median_arr_delay']:.2f} h)\n")
    if not np.isnan(m["wtd_arr_delay"]):
        w(f"  CVM-weighted delay    : {m['wtd_arr_delay']:.2f} h\n")
    w(f"  Preferred scenario %  : {m['preferred_pct']:.1f}%  (1-1 + Multi-1)\n")

    # --- Rule Set 1: flight quality ---
    w(f"\n  {THIN}\n")
    w("  FLIGHT QUALITY  (Rule Set 1)\n")
    w(f"  {THIN}\n")
    w(
        f"  Mean quality score    : {m['mean_quality']:.1f}"
        "  (max ~180 without equipment data)\n"
    )
    for g in ["A", "B", "C", "D"]:
        cnt = m["grades"].get(g, 0)
        pct = 100 * cnt / len(df) if len(df) else 0
        bar = "#" * int(pct / 2)
        w(f"    {g}: {cnt:6d}  ({pct:5.1f}%)  {bar}\n")
    w(f"  Ineligible (>72 h)    : {m['ineligible']}\n")

    w(f"\n  Score components (mean pts per leg):\n")
    w(f"    Arrival delay  : {df['score_arr_delay'].mean():6.1f}\n")
    w(f"    Departure delay: {df['score_dep_delay'].mean():6.1f}\n")
    w(f"    City pair      : {df['score_city_pair'].mean():6.1f}  ")
    cp_pct = 100 * df["city_pair_match"].mean() if "city_pair_match" in df.columns else 0
    w(f"({cp_pct:.1f}% same O&D)\n")
    w(f"    Stopover       : {df['score_stopover'].mean():6.1f}  ")
    direct_pct = 100 * df["IS_DIRECT"].astype(bool).mean()
    w(f"({direct_pct:.1f}% direct)\n")

    # --- Rule Set 3: cabin compliance ---
    w(f"\n  {THIN}\n")
    w("  CABIN COMPLIANCE  (Rule Set 3)\n")
    w(f"  {THIN}\n")
    w(f"  Compliance rate       : {m['cabin_ok_pct']:.1f}%\n")
    for k, v in df["cabin_change"].value_counts().items():
        pct = 100 * v / len(df)
        w(f"    {k:20s}: {v:6d}  ({pct:5.1f}%)\n")

    # --- Journey scenarios ---
    w(f"\n  {THIN}\n")
    w("  JOURNEY SCENARIOS\n")
    w(f"  {THIN}\n")
    order = [
        "1-1 (One-One)",
        "Multi-1 (Multi-One)",
        "1-Multi (One-Multi)",
        "Multi-Multi",
    ]
    for k in order:
        v = m["scenarios"].get(k, 0)
        pct = 100 * v / len(df) if len(df) else 0
        tag = "^ preferred" if "1-1" in k or "Multi-1" in k else "v lower"
        w(f"    {k:26s}: {v:6d}  ({pct:5.1f}%)  {tag}\n")

    w(f"\n{SEP}\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_post_analysis(assignments_df, unbooked_df, available_flights):
    """Run the full Phase-2 rule-set analysis and return a text report.

    Parameters
    ----------
    assignments_df : pandas or polars DataFrame
        Output assignments from the pipeline.
    unbooked_df : pandas or polars DataFrame (or None)
        Unbooked passengers from the pipeline.
    available_flights : str, Path, or DataFrame
        Either a file path to the available-flights CSV, or an already-loaded
        pandas / polars DataFrame.

    Returns
    -------
    str
        Formatted multi-line report.
    """
    # Normalise inputs to pandas
    df = _to_pandas(assignments_df)
    df = _strip_cols(df)

    if unbooked_df is not None:
        unb = _to_pandas(unbooked_df)
        unb = _strip_cols(unb)
    else:
        unb = None

    if isinstance(available_flights, (str, Path)):
        avail = _strip_cols(pd.read_csv(available_flights))
    else:
        avail = _strip_cols(_to_pandas(available_flights))

    # Validate required columns
    missing_asgn = [c for c in ASGN_REQUIRED if c not in df.columns]
    missing_avail = [c for c in AVAIL_REQUIRED if c not in avail.columns]
    if missing_asgn or missing_avail:
        lines = ["Post-analysis could not run — missing columns:"]
        if missing_asgn:
            lines.append(f"  assignments: {missing_asgn}")
        if missing_avail:
            lines.append(f"  available_flights: {missing_avail}")
        return "\n".join(lines)

    # Parse datetime columns
    for col in ["DEP_DTML", "ARR_DTML", "ALT_DEP_DTML", "ALT_ARR_DTML"]:
        if col in df.columns:
            df[col] = _parse_dt(df[col])

    # Run scoring pipeline
    ob = _check_overbooking(df, avail)
    df = _annotate_overbooking(df, ob)
    df = _score_flight_quality(df)
    df = _score_cabin_compliance(df)
    df = _classify_journey(df)
    m = _compute_metrics(df, ob, unb)

    return _format_report(df, ob, m, unb)