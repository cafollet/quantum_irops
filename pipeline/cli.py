"""Command-line interface for the re-accommodation pipeline.

Usage:
    python -m pipeline --pnr pnr.csv --cancelled target.csv --available avail.csv
"""

import argparse

from .runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Airline Passenger Re-accommodation QUBO Pipeline"
    )
    parser.add_argument("--pnr", required=True, help="Path to PNR CSV")
    parser.add_argument("--cancelled", required=True, help="Path to cancelled flights CSV")
    parser.add_argument("--available", required=True, help="Path to available flights CSV")

    parser.add_argument("--method", default="sa", choices=["sa", "neal", "dwave"])
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--t-init", type=float, default=200.0)
    parser.add_argument("--alpha", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--filter-level",
        default="moderate",
        choices=["minimal", "moderate", "aggressive", "ultra"],
    )
    parser.add_argument(
        "--batch-strategy",
        default="auto",
        choices=[
            "none",
            "by_route",
            "by_time_window",
            "by_cabin",
            "by_route_and_time",
            "by_priority_tier",
            "auto",
        ],
    )
    parser.add_argument("--time-window-before", type=float, default=2.0)
    parser.add_argument("--time-window-after", type=float, default=6.0)
    parser.add_argument("--same-cabin-only", action="store_true")
    parser.add_argument("--max-qubo-vars", type=int, default=5000)

    parser.add_argument("--enable-multi-leg", action="store_true")
    parser.add_argument("--max-legs", type=int, default=2)
    parser.add_argument("--min-connection-mins", type=float, default=45.0)
    parser.add_argument("--max-connection-mins", type=float, default=360.0)
    parser.add_argument("--max-itineraries-per-pax", type=int, default=15)
    parser.add_argument(
        "--multi-leg-only-when-no-direct", action="store_true", default=True
    )

    parser.add_argument("--include-non-affected", action="store_true")
    parser.add_argument("--max-non-affected", type=int, default=500)

    parser.add_argument(
        "--priority-bins",
        type=int,
        default=4,
        help=(
            "Number of equal-population CVM quantile bins for "
            "by_priority_tier batching (default: 4)."
        ),
    )
    parser.add_argument(
        "--priority-tiers",
        type=float,
        nargs="+",
        default=None,
        metavar="CVM",
        help=(
            "Manual CVM split points (ascending). Overrides --priority-bins. "
            "E.g. --priority-tiers 2.0 5.0 9.0 → 4 bins."
        ),
    )

    parser.add_argument("--output-assignments", default="assignments.csv")
    parser.add_argument("--output-unbooked", default="unbooked.csv")

    args = parser.parse_args()

    run_pipeline(
        pnr=args.pnr,
        cancelled=args.cancelled,
        available=args.available,
        method=args.method,
        num_reads=args.num_reads,
        t_init=args.t_init,
        alpha=args.alpha,
        seed=args.seed,
        filter_level=args.filter_level,
        batch_strategy=args.batch_strategy,
        time_window_before=args.time_window_before,
        time_window_after=args.time_window_after,
        same_cabin_only=args.same_cabin_only,
        max_qubo_vars=args.max_qubo_vars,
        enable_multi_leg=args.enable_multi_leg,
        max_legs=args.max_legs,
        min_connection_mins=args.min_connection_mins,
        max_connection_mins=args.max_connection_mins,
        max_itineraries_per_pax=args.max_itineraries_per_pax,
        multi_leg_only_when_no_direct=args.multi_leg_only_when_no_direct,
        include_non_affected=args.include_non_affected,
        max_non_affected=args.max_non_affected,
        priority_bins=args.priority_bins,
        priority_tiers=args.priority_tiers,
        output_assignments=args.output_assignments,
        output_unbooked=args.output_unbooked,
        print_summary=True,
    )


if __name__ == "__main__":
    main()
