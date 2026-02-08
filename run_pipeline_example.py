import pipeline
from pipeline import run_pipeline
import logging

# Comment this out when you don't want logging info
logging.basicConfig(level=logging.INFO)

assignments, unbooked = run_pipeline(
    pnr="notebooks/data/PRMI_DM_ALL_PNRs.csv",
    method="neal",
    cancelled="notebooks/data/PRMI-DM_TARGET_FLIGHTS.csv",
    available="notebooks/data/PRMI-DM-AVAILABLE_FLIGHTS.csv",
    batch_strategy="by_priority_tier",
    priority_bins=30,
    output_unbooked="output_files/priority_batch_unbooked_updated.csv",
    output_assignments="output_files/priority_batch_results_updated.csv",
)
