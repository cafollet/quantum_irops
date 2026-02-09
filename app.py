from shiny.express import ui, render
from shinywidgets import render_plotly
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import airportsdata
from utils import get_data_frames

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
            passengers_affected = df_pnrs.filter(pl.col("Affected") == 1)

            passengers_affected = (
                passengers_affected.with_columns(
                    pl.col("CABIN_CD")
                    .replace({"Y": "Economy", "C": "Business"})
                    .alias("Cabin Class")
                )
                .group_by(["DEP_DT", "Cabin Class"])
                .agg(pl.col("PAX_CNT").sum().alias("Total_Passengers"))
                .sort("DEP_DT")
            )

            fig = px.bar(
                passengers_affected,
                x="DEP_DT",
                y="Total_Passengers",
                color="Cabin Class",
                # -- STACKING ORDER --
                # This tells Plotly: Put Economy at the bottom, Business on top.
                category_orders={"Cabin Class": ["Economy", "Business"]},
                # Custom colors (Business = Blue, Economy = Orange)
                color_discrete_map={"Business": "#d3462d", "Economy": "#363636"},
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


## Map plot ##
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
