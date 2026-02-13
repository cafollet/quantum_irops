"""Main pipeline orchestrator and public API function.

ReaccommodationPipeline wires together DataProcessor -> PreprocessingEngine
-> QUBOFormulator -> QUBOSolver -> SolutionInterpreter and handles batch
iteration with live capacity tracking.

run_pipeline() is the single callable entry point for external callers.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

from .candidates import PreprocessingEngine
from .config import MultiLegConfig, PreprocessingConfig, QUBOWeights
from .data import DataProcessor
from .qubo import QUBOFormulator, QUBOSolver
from .results import CapacityTracker, SolutionInterpreter
from .types import BatchStrategy, CandidateFilterLevel

logger = logging.getLogger(__name__)

# Composite key that uniquely identifies one passenger segment.
# A RECLOC alone is not unique — the same booking reference can contain
# multiple legs (e.g. outbound KHL->TPH and return TPH->KHL).
_PNR_KEY = ["RECLOC", "DEP_KEY"]


class ReaccommodationPipeline:
    def __init__(self, weights=None, preprocessing=None):
        self.weights: QUBOWeights = weights or QUBOWeights()
        self.pp_config: PreprocessingConfig = preprocessing or PreprocessingConfig()
        self.processor = DataProcessor()
        self.preprocessor: Optional[PreprocessingEngine] = None
        self.batch_results: list = []
        self.all_assignments: Optional[pd.DataFrame] = None
        self.all_unbooked: Optional[pd.DataFrame] = None
        self._original_affected_pax: int = 0
        self._pass_stats: list = []
        self._final_flights = None

    @staticmethod
    def _to_dataframe(src, name: str) -> pd.DataFrame:
        """Accept either a file path (str/Path) or an already-loaded DataFrame."""
        if isinstance(src, pd.DataFrame):
            return src
        try:
            return pd.read_csv(src)
        except Exception as exc:
            raise ValueError(
                f"Could not load '{name}': expected a file path or DataFrame, "
                f"got {type(src).__name__}. Original error: {exc}"
            ) from exc

    def run(
        self,
        pnr_csv,
        target_csv,
        available_csv,
        method: str = "sa",
        max_passes: int = 1,
        **solver_kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full re-accommodation pipeline, optionally iterating over
        unbooked passengers with updated flight capacity.

        Parameters
        ----------
        pnr_csv, target_csv, available_csv : str | Path | pd.DataFrame
        method : {'sa', 'neal', 'dwave'}
        max_passes : int
            Maximum number of full solve-cycles.  Each pass takes the
            passengers left unbooked by the previous pass and re-runs them
            against the remaining flight capacity.
            Stops early when no new assignments are found or all passengers
            are placed.
        **solver_kwargs : passed through to the chosen solver backend.
        """
        # Reset per-run state
        self._pass_stats = []
        self._final_flights = None

        logger.info("Loading input data...")
        pnr_df = self._to_dataframe(pnr_csv, "pnr_csv")
        target_df = self._to_dataframe(target_csv, "target_csv")
        available_df = self._to_dataframe(available_csv, "available_csv")
        self.processor.load_data(pnr_df, target_df, available_df)

        if self.pp_config.multi_leg.enable_multi_leg:
            logger.warning(
                "MULTI-LEG ITINERARIES ENABLED (max_legs=%d). "
                "This increases QUBO complexity significantly!",
                self.pp_config.multi_leg.max_legs,
            )

        # Store for summary()
        self._original_affected_pax = sum(
            p.pax_cnt for p in self.processor.affected_passengers
        )

        # Accumulate assignments across passes
        # unbooked tracks whos still waiting after pass
        accumulated_assignments: List[pd.DataFrame] = []
        last_unbooked: pd.DataFrame = pd.DataFrame()

        for pass_num in range(1, max_passes + 1):
            n_remaining = len(self.processor.affected_passengers)
            if n_remaining == 0:
                logger.info("Pass %d: no passengers remaining — done.", pass_num)
                break

            logger.info(
                "\n========== PASS %d / %d  —  %d segments (%d PAX) ==========",
                pass_num,
                max_passes,
                n_remaining,
                sum(p.pax_cnt for p in self.processor.affected_passengers),
            )

            cap_tracker = CapacityTracker(self.processor.available_flights)

            pass_assignments, pass_unbooked = self._run_pass(
                cap_tracker, method, **solver_kwargs
            )

            n_new = (
                pass_assignments[_PNR_KEY].drop_duplicates().shape[0]
                if len(pass_assignments)
                else 0
            )
            n_new_pax = (
                int(pass_assignments["PAX_CNT"].sum())
                if len(pass_assignments) and "PAX_CNT" in pass_assignments.columns
                else 0
            )
            n_still_unbooked = (
                int(pass_unbooked["PAX_CNT"].sum())
                if len(pass_unbooked) and "PAX_CNT" in pass_unbooked.columns
                else len(pass_unbooked)
            )
            cumulative_assigned = sum(s["assigned_pax"] for s in self._pass_stats) + n_new_pax
            self._pass_stats.append({
                "pass": pass_num,
                "segments_in": n_remaining,
                "assigned_segments": n_new,
                "assigned_pax": n_new_pax,
                "unbooked_pax": n_still_unbooked,
                "cumulative_pax": cumulative_assigned,
            })
            logger.info(
                "Pass %d: %d new segment(s) assigned, %d still unbooked",
                pass_num,
                n_new,
                len(pass_unbooked) if len(pass_unbooked) else 0,
            )

            if n_new == 0:
                logger.info(
                    "No new assignments in pass %d — stopping early.", pass_num
                )
                last_unbooked = pass_unbooked
                break

            accumulated_assignments.append(pass_assignments)
            last_unbooked = pass_unbooked

            if pass_num < max_passes and len(pass_unbooked):
                # Update available_flights
                self.processor.available_flights = cap_tracker.update_flights(
                    self.processor.available_flights
                )
                # still-unbooked passengers
                unbooked_keys = set(
                    zip(pass_unbooked["RECLOC"], pass_unbooked["DEP_KEY"])
                )
                self.processor.affected_passengers = [
                    p
                    for p in self.processor.affected_passengers
                    if (p.recloc, p.dep_key) in unbooked_keys
                ]

        if accumulated_assignments:
            combined = pd.concat(accumulated_assignments, ignore_index=True)
            # Defensive dedup: keep first assignment per (RECLOC, DEP_KEY)
            self.all_assignments = combined.drop_duplicates(
                subset=_PNR_KEY, keep="first"
            ).reset_index(drop=True)
        else:
            self.all_assignments = pd.DataFrame()

        self.all_unbooked = last_unbooked
        self._final_flights = list(self.processor.available_flights)

        return self.all_assignments, self.all_unbooked

    def _run_pass(
        self,
        cap_tracker: CapacityTracker,
        method: str,
        **solver_kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run one full prepare→batch→solve cycle.
        """
        self.preprocessor = PreprocessingEngine(
            self.processor, self.pp_config, self.weights
        )
        batches = self.preprocessor.prepare()

        dep_key_to_idx = {
            f.dep_key: j for j, f in enumerate(self.processor.available_flights)
        }
        self.batch_results = []

        for batch in batches:
            logger.info("\n--- Solving: %s ---", batch["batch_id"])

            batch["flights"] = cap_tracker.update_flights(
                self.processor.available_flights
            )

            self.preprocessor.rebuild_batch_itineraries(batch, batch["flights"])

            formulator = QUBOFormulator(
                passengers=batch["passengers"],
                flights=batch["flights"],
                itineraries=batch["itineraries"],
                connection_groups=batch["connection_groups"],
                weights=self.weights,
                pp_config=self.pp_config,
            )
            Q = formulator.build()

            if method == "sa":
                sol = QUBOSolver.solve_simulated_annealing(
                    Q, formulator.n_vars, **solver_kwargs
                )
            elif method == "neal":
                sol = QUBOSolver.solve_neal(Q, **solver_kwargs)
            elif method == "dwave":
                sol = QUBOSolver.solve_dwave(Q, **solver_kwargs)
            else:
                raise ValueError(f"Unknown solver method: {method!r}")

            interpreter = SolutionInterpreter(
                passengers=batch["passengers"],
                flights=batch["flights"],
                itineraries=batch["itineraries"],
                index_to_var=formulator.get_reverse_map(),
            )
            adf, udf = interpreter.interpret(sol)
            self.batch_results.append((adf, udf))

            if len(adf):
                seen_segment_flight: set = set()
                for _, row in adf.iterrows():
                    alt_dep_key = row.get("ALT_DEP_KEY", "")
                    cabin = row.get("ALT_CABIN_CD", "")
                    pax_cnt = row.get("PAX_CNT", 0)
                    flt_idx = dep_key_to_idx.get(alt_dep_key)
                    if flt_idx is not None and cabin and pax_cnt:
                        key = (
                            row.get("RECLOC"),
                            row.get("DEP_KEY"),
                            alt_dep_key,
                            cabin,
                        )
                        if key not in seen_segment_flight:
                            seen_segment_flight.add(key)
                            cap_tracker.consume(flt_idx, cabin, int(pax_cnt))

        return self._merge_pass_results()

    def _merge_pass_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merge batch_results for the current pass into single DataFrames."""
        all_a = [a for a, _ in self.batch_results if len(a)]
        all_u = [u for _, u in self.batch_results if len(u)]

        assignments = (
            pd.concat(all_a, ignore_index=True) if all_a else pd.DataFrame()
        )
        unbooked = (
            pd.concat(all_u, ignore_index=True) if all_u else pd.DataFrame()
        )

        if len(assignments):
            if "_itin_id" in assignments.columns:
                # Keep only the first itinerary per (RECLOC, DEP_KEY) segment
                first_itin = (
                    assignments
                    .drop_duplicates(subset=_PNR_KEY, keep="first")[
                        _PNR_KEY + ["_itin_id"]
                    ]
                    .rename(columns={"_itin_id": "_first_itin_id"})
                )
                merged = assignments.merge(first_itin, on=_PNR_KEY, how="left")
                mask = merged["_itin_id"] == merged["_first_itin_id"]
                dropped = int((~mask).sum())
                if dropped:
                    logger.warning(
                        "%d over-assigned row(s) removed during merge "
                        "(solver constraint violations or cross-batch duplicates).",
                        dropped,
                    )
                assignments = (
                    assignments[mask]
                    .drop(columns=["_itin_id"])
                    .reset_index(drop=True)
                )
            else:
                assignments = assignments.drop_duplicates(
                    subset=_PNR_KEY + ["ALT_DEP_KEY", "ALT_CABIN_CD"],
                    keep="first",
                )

        if len(unbooked) and len(assignments):
            booked = assignments[_PNR_KEY].drop_duplicates()
            unbooked = (
                unbooked
                .merge(booked.assign(_b=True), on=_PNR_KEY, how="left")
                .pipe(lambda df: df[df["_b"].isna()])
                .drop(columns=["_b"])
                .reset_index(drop=True)
            )

        return assignments, unbooked

    def export_remaining_flights(self, path: Optional[str] = None) -> pd.DataFrame:
        """Return (and optionally save) the available-flights CSV with updated
        seat counts reflecting everything consumed during this run.

        The returned DataFrame has the same schema as the original
        available_csv so it can be fed directly as ``available`` to a future
        ``run()`` call, enabling external persistence / resume.

        Parameters
        ----------
        path : str | None
            If given, the DataFrame is also written to this CSV path.
        """
        if self._final_flights is None:
            raise RuntimeError(
                "No flight data to export — call run() first."
            )
        rows = []
        for f in self._final_flights:
            rows.append({
                "DEP_KEY": f.dep_key,
                "DEP_DT": f.dep_dt,
                "ORIG_CD": f.orig_cd,
                "DEST_CD": f.dest_cd,
                "FLT_NUM": f.flt_num,
                "DEP_DTML": f.dep_dtml,
                "ARR_DTML": f.arr_dtml,
                "DEP_DTMZ": f.dep_dtmz,
                "ARR_DTMZ": f.arr_dtmz,
                "C_CAP_CNT": f.c_cap_cnt,
                "C_AUL_CNT": f.c_aul_cnt,
                "C_PAX_CNT": f.c_pax_cnt,
                "C_AVAIL_CNT": f.c_avail_cnt,
                "Y_CAP_CNT": f.y_cap_cnt,
                "Y_AUL_CNT": f.y_aul_cnt,
                "Y_PAX_CNT": f.y_pax_cnt,
                "Y_AVAIL_CNT": f.y_avail_cnt,
            })
        df = pd.DataFrame(rows)
        if path:
            df.to_csv(path, index=False)
            logger.info("Remaining flight capacity written to %s", path)
        return df

    def summary(self):
        total_pax = self._original_affected_pax
        booked = 0
        if (
            self.all_assignments is not None
            and len(self.all_assignments)
            and "IS_AFFECTED" in self.all_assignments.columns
        ):
            booked = int(
                self.all_assignments[self.all_assignments["IS_AFFECTED"]][
                    "PAX_CNT"
                ].sum()
            )

        unbooked = int(
            self.all_unbooked["PAX_CNT"].sum()
            if self.all_unbooked is not None and len(self.all_unbooked)
            else 0
        )

        print("\n" + "=" * 70)
        print("RE-ACCOMMODATION SUMMARY")
        print("=" * 70)
        print(f"  Total affected passengers:        {total_pax:,}")
        print(f"  Successfully re-accommodated:     {booked:,}")
        print(f"  Left unbooked:                    {unbooked:,}")
        print(f"  Rate:                             {booked / max(total_pax, 1) * 100:.1f}%")
        print(f"  Batches solved (last pass):       {len(self.batch_results)}")

        if self._pass_stats:
            print(f"\n  Per-pass breakdown:")
            print(f"    {'Pass':<6} {'Segs in':>9} {'New segs':>10} {'New PAX':>9} {'Unbooked':>10} {'Cumul PAX':>11} {'Rate':>7}")
            print(f"    {'-'*60}")
            for s in self._pass_stats:
                rate = s["cumulative_pax"] / max(total_pax, 1) * 100
                print(
                    f"    {s['pass']:<6} {s['segments_in']:>9,} {s['assigned_segments']:>10,} "
                    f"{s['assigned_pax']:>9,} {s['unbooked_pax']:>10,} "
                    f"{s['cumulative_pax']:>11,} {rate:>6.1f}%"
                )

        if self.preprocessor:
            s = self.preprocessor.get_stats()
            print(f"\n  Preprocessing (last pass):")
            print(f"    Strategy:                       {s.get('batch_strategy', 'N/A')}")
            print(f"    Direct itineraries:             {s.get('direct_itineraries', 0)}")
            print(f"    Multi-leg itineraries:          {s.get('multi_leg_itineraries', 0)}")
            print(f"    Pax needing multi-leg:          {s.get('passengers_needing_multi_leg', 0)}")
            print(f"    Pax with no options:            {s.get('passengers_with_no_options', 0)}")
            print(f"    Estimated QUBO vars:            {s.get('estimated_total_vars', 0)}")
            if s.get("priority_tier_info"):
                print(f"\n  Priority Tier Breakdown (last pass, highest → lowest CVM):")
                for t in s["priority_tier_info"]:
                    print(
                        f"    {t['label']}  CVM [{t['cvm_lo']:.3f}, {t['cvm_hi']:.3f}]"
                        f"  {t['pax_count']} pax"
                    )

        if self.all_assignments is not None and len(self.all_assignments):
            a = self.all_assignments
            print(f"\n  Assignment Quality (all passes combined):")
            direct = a[a["IS_DIRECT"]].shape[0] if "IS_DIRECT" in a else 0
            multi = a[~a["IS_DIRECT"]].shape[0] if "IS_DIRECT" in a else 0
            print(f"    Direct assignments:             {direct:,}")
            print(f"    Multi-leg assignments:          {multi:,}")
            if "DEP_CHANGE_MINS" in a:
                print(
                    f"    Avg departure shift (mins):     {a['DEP_CHANGE_MINS'].mean():.0f}"
                )
            if "ITINERARY_CABINS" in a and "ORIG_CABIN" in a:
                cabin_changed = a[a["ORIG_CABIN"] != a["ITINERARY_CABINS"]].shape[0]
                print(f"    Cabin class changes:            {cabin_changed:,}")
            if "IS_AFFECTED" in a:
                non_aff = a[~a["IS_AFFECTED"]].shape[0]
                print(f"    Non-affected moved:             {non_aff:,}")

        print("=" * 70)


def run_pipeline(
    pnr,
    cancelled,
    available,
    # --- solver ---
    method: str = "sa",
    num_reads: int = 100,
    t_init: float = 200.0,
    alpha: float = 0.998,
    seed: int = 42,
    max_passes: int = 1,
    # --- preprocessing ---
    filter_level: str = "moderate",
    batch_strategy: str = "auto",
    time_window_before: float = 2.0,
    time_window_after: float = 6.0,
    time_window_fallback: Optional[List[float]] = None,
    same_cabin_only: bool = False,
    max_qubo_vars: int = 5000,
    # --- multi-leg ---
    enable_multi_leg: bool = False,
    max_legs: int = 2,
    min_connection_mins: float = 45.0,
    max_connection_mins: float = 360.0,
    max_itineraries_per_pax: int = 15,
    multi_leg_only_when_no_direct: bool = True,
    # --- non-affected passengers ---
    include_non_affected: bool = False,
    max_non_affected: int = 500,
    # --- priority tier batching ---
    priority_bins: int = 4,
    priority_tiers: Optional[List[float]] = None,
    # --- power-user overrides ---
    weights: Optional[QUBOWeights] = None,
    preprocessing: Optional[PreprocessingConfig] = None,
    # --- optional output paths ---
    output_assignments: Optional[str] = None,
    output_unbooked: Optional[str] = None,
    output_remaining_flights: Optional[str] = None,
    print_summary: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the passenger re-accommodation QUBO pipeline.

    Parameters
    ----------
    pnr : str | Path | pd.DataFrame
        PNR data (CSV path or pre-loaded DataFrame).
    cancelled : str | Path | pd.DataFrame
        Cancelled/target flights.
    available : str | Path | pd.DataFrame
        Available alternative flights.
    method : {'sa', 'neal', 'dwave'}
        Solver backend.
    num_reads : int
        Annealing restarts / samples.
    t_init : float
        Initial SA temperature.
    alpha : float
        SA cooling rate.
    seed : int
        RNG seed.
    max_passes : int
        How many full solve-cycles to run.  After each pass the remaining
        unbooked passengers are re-queued against the updated flight capacity.
        Stops early when no new assignments are made or all
        passengers are placed
    filter_level : {'minimal', 'moderate', 'aggressive', 'ultra'}
    batch_strategy : {'none','by_route','by_time_window','by_cabin',
                      'by_route_and_time','by_priority_tier','auto'}
    time_window_before : float
        Hours before cancelled departure to search.
    time_window_after : float
        Primary hours after cancelled departure to search.
    time_window_fallback : list[float] | None
        Fallback windows if primary finds nothing. Default [12, 24, 48].
    same_cabin_only : bool
    max_qubo_vars : int
    enable_multi_leg : bool
    max_legs : int
    min_connection_mins : float
    max_connection_mins : float
    max_itineraries_per_pax : int
    multi_leg_only_when_no_direct : bool
    include_non_affected : bool
    max_non_affected : int
    priority_bins : int
        Number of equal-population CVM quantile bins when using
        ``batch_strategy='by_priority_tier'``. Highest-CVM bin is solved
        first so premium passengers get the best seat availability.
        Ignored when ``priority_tiers`` is set.
    priority_tiers : list[float] | None
        Manual CVM split points (ascending). If provided, overrides
        ``priority_bins``. E.g. ``[2.0, 5.0, 9.0]`` -> 4 bins.
    weights : QUBOWeights | None
        Full weight override.
    preprocessing : PreprocessingConfig | None
        Full preprocessing config override.
    output_assignments : str | None
        Save assignments CSV to this path.
    output_unbooked : str | None
        Save unbooked CSV to this path.
    output_remaining_flights : str | None
        Save the updated available-flights CSV
        to this path.
    print_summary : bool

    Returns
    -------
    assignments : pd.DataFrame
    unbooked : pd.DataFrame
    """
    if time_window_fallback is None:
        time_window_fallback = [12.0, 24.0, 48.0]
    if priority_tiers is None:
        priority_tiers = []

    if preprocessing is None:
        ml_config = MultiLegConfig(
            enable_multi_leg=enable_multi_leg,
            max_legs=max_legs,
            min_connection_time_mins=min_connection_mins,
            max_connection_time_mins=max_connection_mins,
            max_itineraries_per_passenger=max_itineraries_per_pax,
            only_when_no_direct=multi_leg_only_when_no_direct,
        )
        preprocessing = PreprocessingConfig(
            filter_level=CandidateFilterLevel(filter_level),
            batch_strategy=BatchStrategy(batch_strategy),
            time_window_before_hours=time_window_before,
            time_window_after_hours=time_window_after,
            time_window_fallback_after_hours=time_window_fallback,
            same_cabin_only=same_cabin_only,
            max_qubo_variables=max_qubo_vars,
            include_non_affected_passengers=include_non_affected,
            max_non_affected_passengers=max_non_affected,
            priority_bins=priority_bins,
            priority_tiers=priority_tiers,
            multi_leg=ml_config,
        )

    if weights is None:
        weights = QUBOWeights()

    pipeline = ReaccommodationPipeline(weights=weights, preprocessing=preprocessing)
    assignments_df, unbooked_df = pipeline.run(
        pnr_csv=pnr,
        target_csv=cancelled,
        available_csv=available,
        method=method,
        max_passes=max_passes,
        num_reads=num_reads,
        T_init=t_init,
        alpha=alpha,
        seed=seed,
    )

    if print_summary:
        pipeline.summary()

    if output_assignments and len(assignments_df):
        assignments_df.to_csv(output_assignments, index=False)
        logger.info("Assignments saved to %s", output_assignments)
    if output_unbooked and len(unbooked_df):
        unbooked_df.to_csv(output_unbooked, index=False)
        logger.info("Unbooked saved to %s", output_unbooked)
    if output_remaining_flights:
        pipeline.export_remaining_flights(output_remaining_flights)

    return assignments_df, unbooked_df