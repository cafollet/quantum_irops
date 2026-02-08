"""Enums and Itinerary value object."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class BatchStrategy(Enum):
    NONE = "none"
    BY_ROUTE = "by_route"
    BY_TIME_WINDOW = "by_time_window"
    BY_CABIN = "by_cabin"
    BY_ROUTE_AND_TIME = "by_route_and_time"
    BY_PRIORITY_TIER = "by_priority_tier"
    AUTO = "auto"


class CandidateFilterLevel(Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA = "ultra"


@dataclass(frozen=True)
class Itinerary:
    """Represents a candidate re-accommodation option.

    Can be a single direct flight or a multi-leg connection.

    legs: tuple of (flight_index, cabin) pairs, ordered by departure.
    Example single leg:  ((3, 'Y'),)
    Example two legs:    ((3, 'Y'), (7, 'Y'))
    """

    legs: Tuple[Tuple[int, str], ...]

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def is_direct(self) -> bool:
        return self.num_legs == 1

    @property
    def flight_indices(self) -> Tuple[int, ...]:
        return tuple(j for j, _ in self.legs)

    @property
    def cabins(self) -> Tuple[str, ...]:
        return tuple(c for _, c in self.legs)

    def __hash__(self):
        return hash(self.legs)

    def __eq__(self, other):
        return isinstance(other, Itinerary) and self.legs == other.legs
