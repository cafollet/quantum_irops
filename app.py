from shiny.express import ui, input, render
from shiny import reactive
from shinywidgets import render_plotly
import polars as pl
import plotly.express as px
import threading
import queue
import time
import random
from shinywidgets import render_plotly
import logging

from pipeline import run_pipeline
from utils import get_data_frames, QueueHandler

canceled_flights = "./notebooks/data/PRMI-DM_TARGET_FLIGHTS.csv"
available_flights = "./notebooks/data/PRMI-DM-AVAILABLE_FLIGHTS.csv"
all_pnrs = "./notebooks/data/PRMI_DM_ALL_PNRs.csv"

df_affected_flights, df_available_flights, df_pnrs = get_data_frames(
    canceled_flights, available_flights, all_pnrs
)
### Dashboard App ###

with ui.div(
    style="background-color: #363636; padding: 15px 20px; margin-bottom: 20px; border-bottom: 2px solid #444;"
):
    ui.h2(
        "IROPS operations Dashboard", style="color: white; margin: 0; font-weight: 400;"
    )

with ui.layout_columns(col_widths=[4, 4, 4]):
    # Affected PNRs pie chart
    with ui.card():
        ui.card_header("Affected PNRs")

        @render_plotly
        def passenger_pie():
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
                # Custom colors: Red for affected, Blue for non-affected
                color_discrete_map={"Affected": "#d3462d", "Non-Affected": "#363636"},
            )
            fig.update_traces(textinfo="percent+label")
            return fig

    # Affected flights pie chart
    with ui.card():
        ui.card_header("Affected Flights")

        @render_plotly
        def flight_pie():
            summary_flight = pl.DataFrame(
                {
                    "Status": ["Non-Affected", "Affected"],
                    "Count": [len(df_available_flights), len(df_affected_flights)],
                }
            )

            fig = px.pie(
                summary_flight,
                values="Count",
                names="Status",
                hole=0.4,
                color="Status",
                # Custom colors: Red for affected, Blue for non-affected
                color_discrete_map={"Affected": "#d3462d", "Non-Affected": "#363636"},
            )
            fig.update_traces(textinfo="percent+label")
            return fig

    with ui.card():
        ui.card_header("Affected Passengers")

        @render_plotly
        def passengers_affected():
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
                # -- STACKING ORDER --
                # This tells Plotly: Put Economy at the bottom, Business on top.
                category_orders={"Cabin Class": ["Economy", "Business"]},
                # Custom colors (Business = Blue, Economy = Orange)
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


########## Map plot #########
def prepare_flight_paths(df_in):

    df_in = df_in.with_columns(
        (pl.col("C_PAX_CNT") + pl.col("Y_PAX_CNT")).alias("Affected_Passengers")
    )

    valid_flights = df_in.drop_nulls(
        subset=["ORIG_LAT", "ORIG_LONG", "DEST_LAT", "DEST_LONG"]
    )

    origins = valid_flights.select(
        pl.col("DEP_KEY"),
        pl.col("ORIG_CD").alias("City_Code"),
        pl.col("ORIG_LAT").alias("latitude"),
        pl.col("ORIG_LONG").alias("longitude"),
        pl.lit("Origin").alias("Type"),
        pl.col("Affected_Passengers"),
    )
    dests = valid_flights.select(
        pl.col("DEP_KEY"),
        pl.col("DEST_CD").alias("City_Code"),
        pl.col("DEST_LAT").alias("latitude"),
        pl.col("DEST_LONG").alias("longitude"),
        pl.lit("Destination").alias("Type"),
        pl.col("Affected_Passengers"),
    )
    return pl.concat([origins, dests], how="vertical")


df_plot = prepare_flight_paths(df_affected_flights).sort("DEP_KEY")

with ui.card(full_screen=True, height="500px"):
    ui.card_header("Affected Flight Routes")

    @render_plotly
    def flight_map():
        fig = px.line_map(
            df_plot,
            lat="latitude",
            lon="longitude",
            line_group="DEP_KEY",
            color_discrete_sequence=["#555555"],
            map_style="carto-darkmatter",
            zoom=1,
            center={"lat": 20, "lon": 0},
        )

        fig_markers = px.scatter_map(
            df_plot,
            lat="latitude",
            lon="longitude",
            size="Affected_Passengers",
            size_max=15,
            color="Type",
            color_discrete_map={
                "Origin": "#d3462d",
                # "Destination": "#2d8ad3",
            },
            hover_name="City_Code",
        )

        fig.add_traces(fig_markers.data)

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_traces(
            selector=dict(
                type="scattermap", mode="lines"
            ),  # Selects only the line layer
            line=dict(width=1),
        )

        return fig


########### Run Optimizer Window ###################


def run_optimization_task(params, msg_queue, shared_state):
    try:
        method = params["method"]
        batch_strategy = params["batch_strategy"]

        queue_handler = QueueHandler(msg_queue)
        queue_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger = logging.getLogger()
        logger.addHandler(queue_handler)
        logger.setLevel(logging.INFO)

        kwargs = {
            "pnr": "notebooks/data/PRMI_DM_ALL_PNRs.csv",
            "method": method,
            "cancelled": "notebooks/data/PRMI-DM_TARGET_FLIGHTS.csv",
            "available": "notebooks/data/PRMI-DM-AVAILABLE_FLIGHTS.csv",
            "batch_strategy": batch_strategy,
            "priority_bins": 5,
            "output_unbooked": "output_files/priority_batch_unbooked_updated.csv",
            "output_assignments": "output_files/priority_batch_results_updated.csv",
        }

        msg_queue.put(f"INFO: Starting {method} optimization...")
        assignments, unbooked = run_pipeline(**kwargs)

        steps = []
        values = []
        current_val = 100.0

        shared_state["result"] = (assignments, unbooked)
        shared_state["status"] = "done"
        msg_queue.put("SUCCESS: Optimization Finished.")

    except Exception as e:
        msg_queue.put(f"ERROR: {str(e)}")
        shared_state["status"] = "error"

    finally:
        logger.removeHandler(queue_handler)


with ui.card(full_screen=True, height="600px"):
    ui.card_header("QUBO Optimizer")

    with ui.layout_columns(col_widths=[4, 8]):
        with ui.div():
            ui.h5("Configuration")

            # Dropdowns for parameters
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

            ui.br()
            ui.input_action_button(
                "btn_run", "Run Optimizer", class_="btn-primary w-100"
            )

        with ui.div():
            # Logging window
            ui.h6("Execution Logs")
            ui.div(
                id="log_container",
                style="height: 400px; overflow-y: auto; background-color: #1e1e1e; padding: 10px; border: 1px solid #555;white-space: pre-wrap;",
            )


session_context = {
    "queue": queue.Queue(),
    "state": {"progress": 0, "status": "idle", "result": None},
}

logs = reactive.Value("")
progress = reactive.Value(0)
is_running = reactive.Value(False)
result_df = reactive.Value(None)


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
        ui=ui.div(
            "--- Process Started ---", style="color: #00ff00; font-family: monospace;"
        ),
    )
    progress.set(0)
    result_df.set(None)
    is_running.set(True)

    session_context["state"] = {"progress": 0, "status": "running", "result": None}
    with session_context["queue"].mutex:
        session_context["queue"].queue.clear()

    params = {
        "method": input.opt_method(),
        "batch_strategy": input.batch_stg(),
    }

    # 4. Launch Thread
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

    try:
        while True:
            # Get one message
            msg = q.get_nowait()

            # CHANGE: Insert immediately instead of appending to a list
            ui.insert_ui(
                selector="#log_container",
                where="beforeEnd",
                ui=ui.div(
                    msg,
                    style="color: #00ff00; font-family: monospace; margin-bottom: 2px;",
                ),
            )
    except queue.Empty:
        pass

    ui.insert_ui(
        selector="#log_container",
        where="beforeEnd",
        ui=ui.tags.script(
            "var d = document.getElementById('log_container'); d.scrollTop = d.scrollHeight;"
        ),
    )
    progress.set(s["progress"])
    status = s["status"]
    if status in ["done", "error"]:
        is_running.set(False)
        if status == "done":
            result_df.set(s["result"])
            ui.notification_show("Optimization Complete!", type="message")
        else:
            ui.notification_show("Optimization Failed", type="error")
            return
    reactive.invalidate_later(20)
