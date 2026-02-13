"""Configuration dataclasses for the re-accommodation pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional

from .types import BatchStrategy, CandidateFilterLevel


@dataclass
class MultiLegConfig:
    """Configuration for multi-leg itinerary construction.

    Only used when enable_multi_leg = True.
    """

    enable_multi_leg: bool = False

    # Maximum number of legs in a connecting itinerary
    max_legs: int = 2

    # Minimum connection time at intermediate airport (minutes)
    min_connection_time_mins: float = 45.0

    # Maximum connection time at intermediate airport (minutes)
    max_connection_time_mins: float = 360.0  # 6 hours

    # Maximum total travel time relative to direct flight time
    # 3.0 means allow up to 3x the original flight duration
    max_travel_time_multiplier: float = 3.0

    # Maximum total travel time absolute cap (minutes)
    max_total_travel_time_mins: float = 1440.0  # 24 hours

    # Only build multi-leg for passengers with NO direct alternatives
    only_when_no_direct: bool = True

    # Maximum itineraries to generate per passenger (cap complexity)
    max_itineraries_per_passenger: int = 15

    # Penalty multiplier for each additional leg
    per_leg_penalty: float = 50.0

    # Intermediate airports to consider (None = all available)
    allowed_hubs: Optional[List[str]] = None


@dataclass
class PreprocessingConfig:
    """All preprocessing knobs to control QUBO size."""

    filter_level: CandidateFilterLevel = CandidateFilterLevel.MODERATE
    time_window_before_hours: float = 2.0
    time_window_after_hours: float = 6.0
    max_candidates_per_passenger: int = 50
    same_cabin_only: bool = False

    # Require at least 1 available seats before a flight enters the
    # candidate pool.
    min_available_seats: int = 1

    batch_strategy: BatchStrategy = BatchStrategy.AUTO
    time_batch_window_hours: float = 4.0

    # Number of equal-population CVM quantile bins for BY_PRIORITY_TIER.
    priority_bins: int = 4

    # Optional manual CVM boundaries (ascending). If non-empty, these
    # override priority_bins and are used directly as bin edges.
    # Example: [2.0, 5.0, 9.0] → 4 bins: (-inf,2], (2,5], (5,9], (9,+inf)
    priority_tiers: List[float] = field(default_factory=list)

    max_qubo_variables: int = 5000
    max_qubo_entries: int = 500000
    target_batch_variables: int = 2000

    # When no candidates exist within the primary window, retry with these
    # progressively wider "after" horizons (hours).
    time_window_fallback_after_hours: List[float] = field(
        default_factory=lambda: [12.0, 24.0, 48.0, 60.0]
    )

    include_non_affected_passengers: bool = False
    non_affected_move_penalty: float = 200.0
    non_affected_same_route_only: bool = True
    max_non_affected_passengers: int = 500

    auto_aggressive_threshold: int = 10000
    auto_batch_threshold: int = 3000

    # Multi-leg config
    multi_leg: MultiLegConfig = field(default_factory=MultiLegConfig)


@dataclass
class QUBOWeights:
    """
    All tunable penalty/reward weights.
    """

    # --- Hard constraints ---
    # one_assignment_penalty must exceed:
    #   (a) max unbooked_passenger_penalty * max_pax_cnt  (so slack=1 beats all-zeros)
    #   (b) capacity_violation_penalty - one_assignment_penalty (so assign-to-full-flight
    #       beats all-zeros for passengers whose only options are full flights).
    # Rule of thumb: one_assignment_penalty ≈ capacity_violation_penalty × 0.6
    # capacity_violation_penalty must still be the single largest penalty so that
    # capacity is never violated in favour of satisfying the one-assignment constraint.
    capacity_violation_penalty: float = 2500000.0
    connection_break_penalty: float = 200000.0
    one_assignment_penalty: float = 1500000.0

    # --- Soft penalties ---
    unbooked_passenger_penalty: float = 8000.0
    departure_time_change_weight: float = 10
    arrival_time_change_weight: float = 10
    cabin_downgrade_penalty: float = 100.0

    # cabin_upgrade_penalty must be large enough to deter Y->C upgrades
    cabin_upgrade_penalty: float = 500.0
    cvm_priority_weight: float = 50.0
    group_size_weight: float = 5.0

    time_window_hours: float = 48.0
    time_window_soft_penalty: float = 2.0

    # Overbooking disabled by default so zero-avail flights are never offered
    # as candidates.  Set to True only if the airline explicitly
    # authorises controlled overbooking and has calibrated the penalty below.
    overbooking_allowed: bool = False
    overbooking_limit_fraction: float = 0.0
    # overbooking_penalty_per_seat is only active when overbooking_allowed=True.
    # Must exceed unbooked_passenger_penalty so overfilling is always more
    # expensive than leaving a passenger unbooked.
    overbooking_penalty_per_seat: float = 2000.0

    non_affected_move_base_penalty: float = 200.0

    # Multi-leg specific
    multi_leg_per_leg_penalty: float = 50.0
    multi_leg_total_time_penalty_weight: float = 0.3

    global_scale: float = 1.0
