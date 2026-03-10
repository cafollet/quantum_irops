from pathlib import Path
from shiny.express import ui, input, render
from shiny import reactive
from shinywidgets import render_plotly
import numpy as np
import polars as pl
import plotly.express as px
import threading
import queue
import time
import re
import logging

from pipeline.config import PreprocessingConfig
from pipeline import run_pipeline
from utils import get_data_frames, QueueHandler
from post_analysis import run_post_analysis


# Default file paths
_DEFAULT_CANCELED = "./notebooks/data/PRMI-DM_TARGET_FLIGHTS.csv"
_DEFAULT_AVAILABLE = "./notebooks/data/PRMI-DM-AVAILABLE_FLIGHTS.csv"
_DEFAULT_PNRS = "./notebooks/data/PRMI_DM_ALL_PNRs.csv"

T = 0.00

ui.include_css(Path(__file__).parent / "styles.css")

########### Dashboard App ######################

with ui.div(class_="app-header"):
    ui.h2("IROPS operations Dashboard")


########### Data source uploads ######################

with ui.card():
    ui.card_header("Data Sources")
    ui.p(
        "Default datasets are loaded automatically. "
        "Upload CSV files below to override any of them.",
        class_="data-source-desc",
    )
    with ui.layout_columns(col_widths=[4, 4, 4]):
        with ui.div():
            ui.input_file("upload_canceled", "Cancelled Flights", accept=[".csv"])
            ui.tags.small(
                f"Default: {_DEFAULT_CANCELED.split('/')[-1]}",
                class_="file-hint",
            )
        with ui.div():
            ui.input_file("upload_available", "Available Flights", accept=[".csv"])
            ui.tags.small(
                f"Default: {_DEFAULT_AVAILABLE.split('/')[-1]}",
                class_="file-hint",
            )
        with ui.div():
            ui.input_file("upload_pnrs", "PNR Data", accept=[".csv"])
            ui.tags.small(
                f"Default: {_DEFAULT_PNRS.split('/')[-1]}",
                class_="file-hint",
            )


########### Reactive data layer ######################

@reactive.calc
def current_file_paths():
    canceled_info = input.upload_canceled()
    available_info = input.upload_available()
    pnrs_info = input.upload_pnrs()
    return {
        "canceled": canceled_info[0]["datapath"] if canceled_info else _DEFAULT_CANCELED,
        "available": available_info[0]["datapath"] if available_info else _DEFAULT_AVAILABLE,
        "pnrs": pnrs_info[0]["datapath"] if pnrs_info else _DEFAULT_PNRS,
    }


@reactive.calc
def current_data():
    paths = current_file_paths()
    return get_data_frames(paths["canceled"], paths["available"], paths["pnrs"])


########### Summary pie charts ######################

with ui.layout_columns(col_widths=[4, 4, 4]):
    with ui.card():
        ui.card_header("Affected PNRs")

        @render_plotly
        def passenger_pie():
            _, _, df_pnrs = current_data()
            summary = df_pnrs.group_by("Affected").len().rename({"len": "count"})
            summary = summary.with_columns(
                pl.when(pl.col("Affected") == 1)
                .then(pl.lit("Affected"))
                .otherwise(pl.lit("Non-Affected"))
                .alias("Status")
            )
            fig = px.pie(
                summary,
                values="count",
                names="Status",
                hole=0.4,
                color="Status",
                color_discrete_map={
                    "Affected": "#d3462d",
                    "Non-Affected": "#363636",
                },
            )
            fig.update_traces(textinfo="percent+label")
            return fig

    with ui.card():
        ui.card_header("Affected Flights")

        @render_plotly
        def flight_pie():
            df_affected, df_available, _ = current_data()
            summary_flight = pl.DataFrame(
                {
                    "Status": ["Non-Affected", "Affected"],
                    "Count": [len(df_available), len(df_affected)],
                }
            )
            fig = px.pie(
                summary_flight,
                values="Count",
                names="Status",
                hole=0.4,
                color="Status",
                color_discrete_map={
                    "Affected": "#d3462d",
                    "Non-Affected": "#363636",
                },
            )
            fig.update_traces(textinfo="percent+label")
            return fig

    with ui.card():
        ui.card_header("Affected Passengers")

        @render_plotly
        def passengers_affected():
            _, _, df_pnrs = current_data()
            def_pass_affected = df_pnrs.filter(pl.col("Affected") == 1)
            def_pass_affected = (
                def_pass_affected.with_columns(
                    pl.col("CABIN_CD")
                    .replace({"Y": "Economy", "C": "Business"})
                    .alias("Cabin Class")
                )
                .group_by(["DEP_DT", "Cabin Class"])
                .agg(pl.col("PAX_CNT").sum().alias("Total_Passengers"))
                .sort("DEP_DT")
            )
            fig = px.bar(
                def_pass_affected,
                x="DEP_DT",
                y="Total_Passengers",
                color="Cabin Class",
                category_orders={"Cabin Class": ["Economy", "Business"]},
                color_discrete_map={"Business": "#A09F9F", "Economy": "#363636"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Departure Date",
                yaxis_title="Passengers",
                legend_title="",
            )
            return fig


########### Map helpers ######################


def _bezier_arc(lat1, lon1, lat2, lon2, n_points=30, curvature=0.15):
    """Generate points along a curved Bezier arc between two coordinates.

    A quadratic Bezier is defined by three points:
        P0 (origin), P1 (control), P2 (destination)

    The control point is offset perpendicular to the straight line
    connecting origin and destination, which creates the visible curve.

    Parameters
    ----------
    curvature : float
        Fraction of the route-vector length to offset the control point.
        Positive curves left (relative to travel direction), negative right.
        Typical range: 0.10 – 0.25.
    """
    t = np.linspace(0, 1, n_points)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Control point: midpoint shifted perpendicular to the route
    ctrl_lat = (lat1 + lat2) / 2 - curvature * dlon
    ctrl_lon = (lon1 + lon2) / 2 + curvature * dlat

    # Quadratic Bezier
    lats = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * ctrl_lat + t**2 * lat2
    lons = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * ctrl_lon + t**2 * lon2

    return lats.tolist(), lons.tolist()


def prepare_map_data(df_in):
    n_total = len(df_in)

    df = df_in.with_columns(
        pl.max_horizontal(
            pl.col("C_PAX_CNT").fill_null(0).cast(pl.Int64) + pl.col("Y_PAX_CNT").fill_null(0).cast(pl.Int64),
            pl.lit(1).cast(pl.Int64),
        ).alias("Pax")
    )

    valid = df.drop_nulls(
        subset=["ORIG_LAT", "ORIG_LONG", "DEST_LAT", "DEST_LONG"]
    )
    n_dropped = n_total - len(valid)

    if len(valid) == 0:
        return None, None, n_dropped

    # aggregate to unique routes
    routes = (
        valid.group_by(["ORIG_CD", "DEST_CD"])
        .agg(
            [
                pl.first("ORIG_LAT").alias("orig_lat"),
                pl.first("ORIG_LONG").alias("orig_lon"),
                pl.first("DEST_LAT").alias("dest_lat"),
                pl.first("DEST_LONG").alias("dest_lon"),
                pl.len().alias("flights"),
                pl.col("Pax").sum().alias("pax"),
            ]
        )
    )

    # aggregate to unique airports
    all_points = pl.concat(
        [
            valid.select(
                pl.col("ORIG_CD").alias("city"),
                pl.col("ORIG_LAT").alias("lat"),
                pl.col("ORIG_LONG").alias("lon"),
                pl.col("Pax").alias("departing_pax"),
                pl.lit(0).cast(pl.Int64).alias("arriving_pax"),  # cast to Int64
            ),
            valid.select(
                pl.col("DEST_CD").alias("city"),
                pl.col("DEST_LAT").alias("lat"),
                pl.col("DEST_LONG").alias("lon"),
                pl.lit(0).cast(pl.Int64).alias("departing_pax"),  # cast to Int64
                pl.col("Pax").alias("arriving_pax"),
            ),
        ]
    )

    airports = (
        all_points.group_by("city")
        .agg(
            [
                pl.first("lat"),
                pl.first("lon"),
                pl.col("departing_pax").sum(),
                pl.col("arriving_pax").sum(),
            ]
        )
        .with_columns(
            [
                (pl.col("departing_pax") + pl.col("arriving_pax")).alias(
                    "total_pax"
                ),
                pl.when(
                    (pl.col("departing_pax") > 0) & (pl.col("arriving_pax") > 0)
                )
                .then(pl.lit("Hub"))
                .when(pl.col("departing_pax") > 0)
                .then(pl.lit("Origin"))
                .otherwise(pl.lit("Destination"))
                .alias("role"),
            ]
        )
    )

    # build curved arc segments
    seg_ids: list[str] = []
    seg_lats: list[float] = []
    seg_lons: list[float] = []
    seg_halves: list[str] = []

    for row in routes.iter_rows(named=True):
        route_id = f"{row['ORIG_CD']}_{row['DEST_CD']}"
        sign = 1.0 if row["ORIG_CD"] < row["DEST_CD"] else -1.0

        lats, lons = _bezier_arc(
            row["orig_lat"],
            row["orig_lon"],
            row["dest_lat"],
            row["dest_lon"],
            n_points=30,
            curvature=0.15 * sign,
        )

        mid = len(lats) // 2

        for i in range(mid + 1):
            seg_ids.append(f"{route_id}__o")
            seg_lats.append(lats[i])
            seg_lons.append(lons[i])
            seg_halves.append("Origin")

        for i in range(mid, len(lats)):
            seg_ids.append(f"{route_id}__d")
            seg_lats.append(lats[i])
            seg_lons.append(lons[i])
            seg_halves.append("Destination")

    segments = pl.DataFrame(
        {
            "segment_id": seg_ids,
            "latitude": seg_lats,
            "longitude": seg_lons,
            "half": seg_halves,
        }
    )

    return airports, segments, n_dropped


########### Map card ######################

with ui.card(full_screen=True, height="500px"):
    ui.card_header("Affected Flight Routes")

    @render_plotly
    def flight_map():
        df_affected, _, _ = current_data()

        # Guard: empty dataset
        if len(df_affected) == 0:
            fig = px.scatter_map(
                lat=[0], lon=[0], map_style="carto-darkmatter", zoom=1
            )
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(
                        text="No affected flights in dataset",
                        showarrow=False,
                        font=dict(color="#888", size=16),
                    )
                ],
            )
            return fig

        airports, segments, n_dropped = prepare_map_data(df_affected)

        # Guard: all flights had missing coordinates
        if airports is None:
            fig = px.scatter_map(
                lat=[0], lon=[0], map_style="carto-darkmatter", zoom=1
            )
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(
                        text=f"All {n_dropped} flights missing coordinates",
                        showarrow=False,
                        font=dict(color="#888", size=16),
                    )
                ],
            )
            return fig

        # curved route arcs (red origin-half → blue dest-half)
        fig = px.line_map(
            segments,
            lat="latitude",
            lon="longitude",
            line_group="segment_id",
            color="half",
            color_discrete_map={
                "Origin": "#d3462d",
                "Destination": "#3498db",
            },
            map_style="carto-darkmatter",
            zoom=1,
            center={"lat": 20, "lon": 0},
        )

        for trace in fig.data:
            trace.legendgroup = trace.name
            trace.showlegend = False
            trace.hoverinfo = "skip"  # airports handle tooltips
        fig.update_traces(line=dict(width=2.5))

        # airport markers (coloured by role)
        fig_airports = px.scatter_map(
            airports,
            lat="lat",
            lon="lon",
            size="total_pax",
            size_max=22,
            color="role",
            color_discrete_map={
                "Origin": "#d3462d",
                "Destination": "#3498db",
                "Hub": "#f39c12",
            },
            hover_name="city",
            hover_data={
                "role": True,
                "total_pax": True,
                "departing_pax": True,
                "arriving_pax": True,
                "lat": False,
                "lon": False,
            },
            labels={
                "total_pax": "Total PAX",
                "departing_pax": "Departing PAX",
                "arriving_pax": "Arriving PAX",
            },
        )
        # Bind marker legend groups so they toggle with matching arcs
        for trace in fig_airports.data:
            trace.legendgroup = trace.name
        fig.add_traces(fig_airports.data)

        # ── subtitle with counts ────────────────────────────────────────
        n_routes = (
            segments.select("segment_id").unique().height // 2
        )  # each route has __o and __d
        n_airports = len(airports)
        subtitle = f"{n_routes} routes across {n_airports} airports"
        if n_dropped:
            subtitle += f" | {n_dropped} flights omitted (missing coordinates)"

        fig.update_layout(
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                bgcolor="rgba(30,30,30,0.8)",
                font=dict(color="white"),
                title=dict(text="Airport Role", font=dict(size=11)),
            ),
            title=dict(
                text=subtitle,
                font=dict(size=11, color="#888"),
                x=0.01,
                y=0.98,
            ),
        )

        return fig


########### Run Optimizer Window ###################


def run_optimization_task(params, msg_queue, shared_state):
    try:
        method = params["method"]
        batch_strategy = params["batch_strategy"]
        num_bins = int(params["num_bins"])
        upgrade_allowed = params["upgrade_allowed"] == "Yes"
        max_passes = int(params["max_passes"])

        queue_handler = QueueHandler(msg_queue)
        queue_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger = logging.getLogger()
        logger.addHandler(queue_handler)
        logger.setLevel(logging.INFO)

        kwargs = {
            "pnr": params["pnr_path"],
            "method": method,
            "cancelled": params["canceled_path"],
            "available": params["available_path"],
            "upgrade_allowed": upgrade_allowed,
            "batch_strategy": batch_strategy,
            "priority_bins": num_bins,
            "time_window_after": 72.0,
            "enable_multi_leg": True,
            "max_legs": 2,
            "max_passes": max_passes,
            "output_unbooked": "output_files/priority_batch_unbooked_updated.csv",
            "output_assignments": "output_files/priority_batch_results_updated.csv",
        }

        msg_queue.put(f"INFO: Starting {method} optimization...")

        t_0 = time.time()
        assignments, unbooked = run_pipeline(**kwargs)
        t_1 = time.time()

        global T
        T = t_1 - t_0

        shared_state["result"] = (assignments, unbooked)
        shared_state["status"] = "done"
        msg_queue.put(
            "SUCCESS: Optimization Finished in {} minutes and {} seconds.".format(
                int(T // 60), int(T % 60)
            )
        )

    except Exception as e:
        msg_queue.put(f"ERROR: {str(e)}")
        shared_state["status"] = "error"

    finally:
        logger.removeHandler(queue_handler)


########### Progress-parsing helpers ###################

_RE_PASS_HEADER = re.compile(
    r"PASS\s+(\d+)\s*/\s*(\d+)\s+.*?(\d+)\s+segments\s*\((\d+)\s*PAX\)"
)
_RE_BATCH_SOLVE = re.compile(r"---\s*Solving:\s*(.+?)\s*---")
_RE_PASS_DONE = re.compile(r"Pass\s+(\d+):\s*(\d+)\s+new segment")

_PREPROCESS_END_PCT = 10.0
_PASS1_WEIGHT_PCT = 60.0
_SUBSEQUENT_WEIGHT_PCT = 30.0


def _make_progress_state(expected_batches: int, max_passes: int) -> dict:
    pass1_end = (
        _PREPROCESS_END_PCT + _PASS1_WEIGHT_PCT
        if max_passes > 1
        else 100.0
    )
    return {
        "phase": "idle",
        "current_pass": 0,
        "max_passes": max_passes,
        "expected_batches": max(expected_batches, 1),
        "batches_done_p1": 0,
        "batches_done_pn": 0,
        "pass1_end_pct": pass1_end,
        "_pn_start": pass1_end,
        "_pn_end": 100.0,
        "pct": 0.0,
        "status_text": "Waiting to start\u2026",
    }


def _parse_progress(msg: str, prog: dict) -> None:
    if "Loading input data" in msg:
        prog["phase"] = "preprocessing"
        prog["pct"] = 2.0
        prog["status_text"] = "Loading input data\u2026"
        return

    if "MULTI-LEG ITINERARIES ENABLED" in msg:
        if prog["phase"] == "preprocessing":
            prog["pct"] = 3.0
            prog["status_text"] = "Configuring multi-leg itineraries\u2026"
        return

    m = _RE_PASS_HEADER.search(msg)
    if m:
        pass_num = int(m.group(1))
        max_passes = int(m.group(2))
        n_segments = m.group(3)
        n_pax = m.group(4)
        prog["current_pass"] = pass_num
        prog["max_passes"] = max_passes
        prog["pass1_end_pct"] = (
            _PREPROCESS_END_PCT + _PASS1_WEIGHT_PCT
            if max_passes > 1
            else 100.0
        )
        if pass_num == 1:
            prog["phase"] = "preprocessing"
            prog["batches_done_p1"] = 0
            prog["pct"] = 5.0
            prog["status_text"] = (
                f"Pass 1/{max_passes}: preparing {n_segments} segments "
                f"({n_pax} PAX)\u2026"
            )
        else:
            prog["phase"] = "pass_n"
            prog["batches_done_pn"] = 0
            n_sub = max(max_passes - 1, 1)
            idx = pass_num - 2
            band_start = prog["pass1_end_pct"]
            band_total = 100.0 - band_start
            prog["_pn_start"] = band_start + band_total * idx / n_sub
            prog["_pn_end"] = band_start + band_total * (idx + 1) / n_sub
            prog["pct"] = prog["_pn_start"]
            prog["status_text"] = (
                f"Pass {pass_num}/{max_passes}: preparing {n_segments} "
                f"segments ({n_pax} PAX)\u2026"
            )
        return

    m = _RE_BATCH_SOLVE.search(msg)
    if m:
        batch_id = m.group(1).strip()
        cp = prog.get("current_pass", 1) or 1
        if cp == 1:
            if prog["phase"] in ("idle", "preprocessing"):
                prog["phase"] = "pass_1"
            prog["batches_done_p1"] += 1
            done = prog["batches_done_p1"]
            total = prog["expected_batches"]
            ratio = min(done / total, 1.0)
            p1_range = prog["pass1_end_pct"] - _PREPROCESS_END_PCT
            prog["pct"] = min(
                _PREPROCESS_END_PCT + ratio * p1_range,
                prog["pass1_end_pct"] - 1.0,
            )
            prog["status_text"] = (
                f"Pass 1 \u2014 batch {done}/{total}: {batch_id}"
            )
        else:
            prog["batches_done_pn"] += 1
            done = prog["batches_done_pn"]
            total = prog["expected_batches"]
            ratio = min(done / max(total, 1), 1.0)
            pn_s = prog["_pn_start"]
            pn_e = prog["_pn_end"]
            prog["pct"] = min(pn_s + ratio * (pn_e - pn_s), pn_e - 1.0)
            prog["status_text"] = (
                f"Pass {cp}/{prog['max_passes']} \u2014 "
                f"batch {done}: {batch_id}"
            )
        return

    m = _RE_PASS_DONE.search(msg)
    if m:
        pass_num = int(m.group(1))
        n_assigned = m.group(2)
        if pass_num == 1:
            prog["pct"] = prog["pass1_end_pct"]
            prog["status_text"] = (
                f"Pass 1 complete \u2014 {n_assigned} segments assigned"
            )
        else:
            prog["pct"] = prog.get("_pn_end", 100.0)
            prog["status_text"] = (
                f"Pass {pass_num} complete \u2014 {n_assigned} segments assigned"
            )
        return

    if "no passengers remaining" in msg:
        prog["pct"] = 100.0
        prog["status_text"] = "All passengers accommodated \u2713"
        return

    if "No new assignments" in msg:
        prog["pct"] = 100.0
        prog["status_text"] = "No further improvements \u2014 done"
        return

    if "SUCCESS" in msg:
        prog["pct"] = 100.0
        prog["phase"] = "done"
        prog["status_text"] = "Optimisation complete \u2713"
        return


### UI — Optimizer card ###################

with ui.card(full_screen=True, height="650px"):
    ui.card_header("QUBO Optimizer")

    with ui.layout_columns(col_widths=[4, 8]):
        with ui.div():
            ui.h5("Configuration")

            ui.input_select(
                "opt_method",
                "Optimization Method",
                choices=["sa", "neal", "dwave"],
                selected="neal",
            )
            ui.input_select(
                "batch_stg",
                "Batch Strategy",
                choices=[
                    "by_route",
                    "by_time_window",
                    "by_cabin",
                    "by_route_and_time",
                    "by_priority_tier",
                    "auto",
                ],
                selected="by_priority_tier",
            )
            ui.input_numeric(
                "num_bins", "Number of bins:", 30, min=1, max=100, step=1
            )
            ui.input_select(
                "upgrade_allowed",
                "Class Upgrade Allowed?",
                choices=["Yes", "No"],
                selected="No",
            )
            ui.input_numeric(
                "max_passes", "Max Passes:", 2, min=1, max=10, step=1
            )

            ui.br()
            ui.input_action_button(
                "btn_run", "Run Optimizer", class_="btn-primary w-100"
            )

        with ui.div(style="display: flex; flex-direction: column; gap: 12px;"):
            ui.h6("Optimizer Progress")

            ui.div(id="progress_status", class_="progress-status")

            with ui.div(
                style="display: flex; align-items: center; gap: 10px;"
            ):
                with ui.div(class_="progress-track"):
                    ui.div(id="progress_bar_fill", class_="progress-fill")
                ui.span("0%", id="progress_pct_text", class_="progress-pct")

            ui.div(id="__progress_js", style="display:none;")

            ui.br()

            with ui.tags.details():
                ui.tags.summary("Execution Logs (debug)")
                ui.div(id="log_container", class_="log-container")


########### Post-Analysis Card ###################

with ui.card():
    ui.card_header("Post-Analysis: Phase 2 Rules Set")

    @render.ui
    def analysis_output():
        report = analysis_report.get()
        if report is None:
            return ui.div(
                ui.p(
                    "Run the optimizer to see rule-set validation results.",
                    style="color: var(--text-secondary); font-style: italic; margin: 10px 0;",
                )
            )
        has_warning = "OVERBOOKING DETECTED" in report
        border_color = "#C0392B" if has_warning else "#27AE60"
        return ui.div(
            ui.pre(
                report,
                style=f"border-left: 4px solid {border_color};",
            )
        )


########### Reactive state ###################

session_context = {
    "queue": queue.Queue(),
    "state": {"progress": 0, "status": "idle", "result": None},
    "prog": _make_progress_state(expected_batches=30, max_passes=2),
    "file_paths": {},
}

logs = reactive.Value("")
progress = reactive.Value(0)
is_running = reactive.Value(False)
result_df = reactive.Value(None)
analysis_report = reactive.Value(None)


def _push_progress_to_ui(prog: dict, bar_color: str = "") -> None:
    pct = prog["pct"]
    text = prog["status_text"]

    ui.remove_ui(selector="#progress_status > *", multiple=True)
    ui.insert_ui(
        selector="#progress_status",
        where="beforeEnd",
        ui=ui.span(text),
    )

    color_js = (
        f"document.getElementById('progress_bar_fill').style.background = '{bar_color}';"
        if bar_color
        else ""
    )
    ui.insert_ui(
        selector="#__progress_js",
        where="beforeEnd",
        ui=ui.tags.script(
            f"document.getElementById('progress_bar_fill').style.width = '{pct:.1f}%';"
            f"document.getElementById('progress_pct_text').textContent = '{pct:.0f}%';"
            + color_js
        ),
    )


@reactive.Effect
@reactive.event(input.btn_run)
def start_optimization():
    if is_running.get():
        ui.notification_show("Optimizer is already running!", type="warning")
        return

    ui.remove_ui(selector="#log_container > *", multiple=True)
    ui.insert_ui(
        selector="#log_container",
        where="beforeEnd",
        ui=ui.div("--- Process Started ---"),
    )

    max_passes = int(input.max_passes())
    session_context["prog"] = _make_progress_state(
        expected_batches=int(input.num_bins()),
        max_passes=max_passes,
    )
    _push_progress_to_ui(session_context["prog"])

    progress.set(0)
    result_df.set(None)
    analysis_report.set(None)
    is_running.set(True)

    session_context["state"] = {"progress": 0, "status": "running", "result": None}
    with session_context["queue"].mutex:
        session_context["queue"].queue.clear()

    paths = current_file_paths()
    session_context["file_paths"] = paths

    params = {
        "method": input.opt_method(),
        "batch_strategy": input.batch_stg(),
        "num_bins": input.num_bins(),
        "upgrade_allowed": input.upgrade_allowed(),
        "max_passes": max_passes,
        "pnr_path": paths["pnrs"],
        "canceled_path": paths["canceled"],
        "available_path": paths["available"],
    }

    t = threading.Thread(
        target=run_optimization_task,
        args=(params, session_context["queue"], session_context["state"]),
        daemon=True,
    )
    t.start()

    reactive.invalidate_later(100)


@reactive.Effect
def poll_background_thread():
    if not is_running.get():
        return

    q = session_context["queue"]
    s = session_context["state"]
    prog = session_context["prog"]

    progress_changed = False

    try:
        while True:
            msg = q.get_nowait()

            ui.insert_ui(
                selector="#log_container",
                where="beforeEnd",
                ui=ui.div(msg, style="margin-bottom: 2px;"),
            )

            old_pct = prog["pct"]
            old_text = prog["status_text"]
            _parse_progress(msg, prog)
            if prog["pct"] != old_pct or prog["status_text"] != old_text:
                progress_changed = True

    except queue.Empty:
        pass

    ui.insert_ui(
        selector="#log_container",
        where="beforeEnd",
        ui=ui.tags.script(
            "var d = document.getElementById('log_container');"
            " d.scrollTop = d.scrollHeight;"
        ),
    )

    if progress_changed:
        _push_progress_to_ui(prog)

    progress.set(s["progress"])
    status = s["status"]

    if status in ["done", "error"]:
        is_running.set(False)

        if status == "done":
            prog["pct"] = 100.0
            prog["phase"] = "done"
            prog["status_text"] = "Optimisation complete \u2713"
            _push_progress_to_ui(prog)

            assignments, unbooked = s["result"]
            result_df.set(s["result"])
            ui.notification_show(
                "Optimization Complete in {} minutes and {} seconds!"
                " Running post-analysis\u2026".format(
                    int(T // 60), int(T % 60)
                ),
                type="message",
            )
            try:
                report = run_post_analysis(
                    assignments_df=assignments,
                    unbooked_df=unbooked,
                    available_flights=session_context["file_paths"]["available"],
                )
                analysis_report.set(report)
            except Exception as exc:
                analysis_report.set(f"Post-analysis error: {exc}")
        else:
            prog["phase"] = "error"
            prog["status_text"] = "Optimisation failed \u2717"
            _push_progress_to_ui(prog, bar_color="#e74c3c")
            ui.notification_show("Optimization Failed", type="error")
        return

    reactive.invalidate_later(1)