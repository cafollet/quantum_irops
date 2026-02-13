from pipeline import run_pipeline
import logging
import time

# Comment this out when you don't want logging info
logging.basicConfig(level=logging.INFO)

t_0 = time.time()

assignments, unbooked = run_pipeline(
    pnr="notebooks/data/PRMI_DM_ALL_PNRs.csv",
    method="neal",
    cancelled="notebooks/data/PRMI-DM_TARGET_FLIGHTS.csv",
    available="notebooks/data/PRMI-DM-AVAILABLE_FLIGHTS.csv",
    batch_strategy="by_priority_tier",
    priority_bins=30,
    time_window_after=72.0,
    enable_multi_leg=True,
    max_legs=2,
    max_passes=2,
    output_unbooked="output_files/priority_batch_unbooked_updated.csv",
    output_assignments="output_files/priority_batch_results_updated.csv",
)

t_1 = time.time()
logging.info(
    "Finished in {} minutes {} seconds".format(
        int((t_1 - t_0) // 60), int((t_1 - t_0) % 60)
    )
)
