"""Candidate itinerary generation and batch preprocessing.

ItineraryBuilder constructs direct and multi-leg alternatives for each
passenger.  PreprocessingEngine wraps it, decides a batching strategy,
and returns batch dicts for the QUBO layer.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np

from .config import MultiLegConfig, PreprocessingConfig, QUBOWeights
from .data import DataProcessor
from .models import Flight, Passenger
from .types import BatchStrategy, CandidateFilterLevel, Itinerary

logger = logging.getLogger(__name__)


class ItineraryBuilder:
    """Constructs candidate itineraries (single and multi-leg) for passengers.

    Multi-leg itineraries are only built when explicitly enabled.
    """

    def __init__(
        self,
        flights: List[Flight],
        ml_config: MultiLegConfig,
        pp_config: PreprocessingConfig,
        weights: QUBOWeights,
    ):
        self.flights = flights
        self.ml = ml_config
        self.pp = pp_config
        self.w = weights

        # Indexes for fast lookup
        self._by_origin: Dict[str, List[int]] = defaultdict(list)
        self._by_route: Dict[str, List[int]] = defaultdict(list)
        self._by_dest: Dict[str, List[int]] = defaultdict(list)
        self._all_airports: Set[str] = set()
        self._build_indexes()

        # Stats for logging
        self.stats = {
            "direct_itineraries": 0,
            "multi_leg_itineraries": 0,
            "passengers_needing_multi_leg": 0,
            "passengers_with_no_options": 0,
        }

    def _build_indexes(self):
        for j, flt in enumerate(self.flights):
            self._by_origin[flt.orig_cd].append(j)
            self._by_dest[flt.dest_cd].append(j)
            self._by_route[flt.route_key].append(j)
            self._all_airports.add(flt.orig_cd)
            self._all_airports.add(flt.dest_cd)

    def build_itineraries(
        self, passengers: List[Passenger]
    ) -> Dict[int, List[Itinerary]]:
        """Build candidate itineraries for each passenger.

        Returns:
            Dict mapping passenger index -> list of Itinerary objects.
        """
        result: Dict[int, List[Itinerary]] = {}

        for i, pax in enumerate(passengers):

            # Build direct itineraries
            direct = self._build_direct_itineraries(pax)
            pax.has_direct_option = len(direct) > 0

            itineraries = list(direct)

            # Build multi-leg if enabled
            if self.ml.enable_multi_leg:
                need_multi = (
                    not self.ml.only_when_no_direct or not pax.has_direct_option
                )
                if need_multi:
                    multi = self._build_multi_leg_itineraries(pax)
                    itineraries.extend(multi)

                    if not pax.has_direct_option and len(multi) > 0:
                        self.stats["passengers_needing_multi_leg"] += 1

            if (
                self.pp.filter_level == CandidateFilterLevel.ULTRA
                and len(itineraries) > self.pp.max_candidates_per_passenger
            ):
                itineraries = self._top_k_itineraries(
                    pax, itineraries, self.pp.max_candidates_per_passenger
                )

            if not itineraries:
                self.stats["passengers_with_no_options"] += 1

            # Count stats
            for itin in itineraries:
                if itin.is_direct:
                    self.stats["direct_itineraries"] += 1
                else:
                    self.stats["multi_leg_itineraries"] += 1

            result[i] = itineraries

        logger.info(
            "Itineraries built: %d direct, %d multi-leg, "
            "%d pax needed multi-leg, %d pax with no options",
            self.stats["direct_itineraries"],
            self.stats["multi_leg_itineraries"],
            self.stats["passengers_needing_multi_leg"],
            self.stats["passengers_with_no_options"],
        )

        return result


    def _build_direct_itineraries(self, pax: Passenger) -> List[Itinerary]:
        """Build single-leg itineraries matching origin-destination routes.

        If the primary time window yields nothing, widen the window
        through the configured fallback steps, until at least one
        candidate is found (or window is maximized, dictated by hard constraint).
        """
        route_flights = self._by_route.get(pax.route_key, [])

        apply_window = self.pp.filter_level.value in (
            "moderate",
            "aggressive",
            "ultra",
        )
        if apply_window:
            windows_to_try = [self.pp.time_window_after_hours] + list(
                self.pp.time_window_fallback_after_hours
            )
        else:
            windows_to_try = [None]  # None = no filter

        for after_hrs in windows_to_try:
            itineraries = []
            for j in route_flights:
                flt = self.flights[j]
                if after_hrs is not None:
                    if not self._within_time_window(pax, flt, after_hrs):
                        continue
                cabins = self._get_candidate_cabins(pax, flt)
                for cabin in cabins:
                    if self._check_capacity(flt, cabin):
                        itineraries.append(Itinerary(legs=((j, cabin),)))
            if itineraries:
                if after_hrs is not None and after_hrs > self.pp.time_window_after_hours:
                    logger.debug(
                        "Pax %s (%s): widened window to +%.0fh, found %d options",
                        pax.recloc,
                        pax.route_key,
                        after_hrs,
                        len(itineraries),
                    )
                return itineraries

        return []

    def _build_multi_leg_itineraries(self, pax: Passenger) -> List[Itinerary]:
        """Build multi-leg connecting itineraries.

        Applies the same adaptive time-window fallback as _build_direct_itineraries
        so thin-route passengers aren't left with zero options.
        """
        if self.ml.max_legs < 2:
            return []

        max_total_mins = self._get_max_travel_time(pax)
        hubs = self._get_candidate_hubs(pax)

        windows_to_try = [self.pp.time_window_after_hours] + list(
            self.pp.time_window_fallback_after_hours
        )

        for after_hrs in windows_to_try:
            itineraries: List[Itinerary] = []

            for hub in hubs:
                if hub == pax.orig_cd or hub == pax.dest_cd:
                    continue

                first_legs = self._by_route.get(f"{pax.orig_cd}-{hub}", [])
                second_legs = self._by_route.get(f"{hub}-{pax.dest_cd}", [])

                if not first_legs or not second_legs:
                    continue

                for j1 in first_legs:
                    flt1 = self.flights[j1]
                    if not self._within_time_window(pax, flt1, after_hrs):
                        continue

                    for j2 in second_legs:
                        flt2 = self.flights[j2]

                        if not self._valid_connection(flt1, flt2):
                            continue
                        if not self._check_total_time(pax, flt1, flt2, max_total_mins):
                            continue

                        for c1 in self._get_candidate_cabins(pax, flt1):
                            for c2 in self._get_candidate_cabins(pax, flt2):
                                if self._check_capacity(
                                    flt1, c1
                                ) and self._check_capacity(flt2, c2):
                                    itineraries.append(
                                        Itinerary(legs=((j1, c1), (j2, c2)))
                                    )

                        if len(itineraries) >= self.ml.max_itineraries_per_passenger:
                            return itineraries[: self.ml.max_itineraries_per_passenger]

            if self.ml.max_legs >= 3 > len(itineraries):
                itineraries.extend(
                    self._build_three_leg(pax, hubs, max_total_mins, after_hrs)
                )

            if itineraries:
                return itineraries[: self.ml.max_itineraries_per_passenger]

        if len(itineraries) > self.ml.max_itineraries_per_passenger:
            itineraries = self._top_k_itineraries(
                pax, itineraries, self.ml.max_itineraries_per_passenger
            )
        return itineraries

    def _build_three_leg(
            self,
            pax: Passenger,
            hubs: Set[str],
            max_total_mins: float,
            after_hrs: float,
    ) -> List[Itinerary]:
        """Build 3-leg itineraries (ORIG -> H1 -> H2 -> DEST)."""
        itineraries = []
        cap = self.ml.max_itineraries_per_passenger
        hub_list = list(hubs - {pax.orig_cd, pax.dest_cd})

        for h1 in hub_list:
            for h2 in hub_list:
                if h1 == h2:
                    continue

                legs1 = self._by_route.get(f"{pax.orig_cd}-{h1}", [])
                legs2 = self._by_route.get(f"{h1}-{h2}", [])
                legs3 = self._by_route.get(f"{h2}-{pax.dest_cd}", [])

                if not legs1 or not legs2 or not legs3:
                    continue

                for j1 in legs1[:5]:
                    flt1 = self.flights[j1]
                    if not self._within_time_window(pax, flt1, after_hrs):
                        continue
                    for j2 in legs2[:5]:
                        flt2 = self.flights[j2]
                        if not self._valid_connection(flt1, flt2):
                            continue
                        for j3 in legs3[:5]:
                            flt3 = self.flights[j3]
                            if not self._valid_connection(flt2, flt3):
                                continue
                            if flt1.dep_dtml and flt3.arr_dtml:
                                total = (
                                    flt3.arr_dtml - flt1.dep_dtml
                                ).total_seconds() / 60.0
                                if total > max_total_mins:
                                    continue
                            cabin = pax.cabin_cd
                            if (
                                self._check_capacity(flt1, cabin)
                                and self._check_capacity(flt2, cabin)
                                and self._check_capacity(flt3, cabin)
                            ):
                                itineraries.append(
                                    Itinerary(
                                        legs=(
                                            (j1, cabin),
                                            (j2, cabin),
                                            (j3, cabin),
                                        )
                                    )
                                )
                            if len(itineraries) >= cap:
                                return itineraries[:cap]

        return itineraries

    def _get_candidate_hubs(self, pax: Passenger) -> Set[str]:
        if self.ml.allowed_hubs is not None:
            return set(self.ml.allowed_hubs)
        from_origin = {
            self.flights[j].dest_cd
            for j in self._by_origin.get(pax.orig_cd, [])
        }
        to_dest = {
            self.flights[j].orig_cd for j in self._by_dest.get(pax.dest_cd, [])
        }
        return from_origin & to_dest

    def _within_time_window(
        self, pax: Passenger, flt: Flight, after_hrs: Optional[float] = None
    ) -> bool:
        if pax.dep_dtml is None or flt.dep_dtml is None:
            return True
        after = (
            after_hrs if after_hrs is not None else self.pp.time_window_after_hours
        )
        delta_hrs = (flt.dep_dtml - pax.dep_dtml).total_seconds() / 3600.0
        return -self.pp.time_window_before_hours <= delta_hrs <= after

    def _valid_connection(self, flt1: Flight, flt2: Flight) -> bool:
        if flt1.arr_dtml is None or flt2.dep_dtml is None:
            return False
        if flt1.dest_cd != flt2.orig_cd:
            return False
        conn_mins = (flt2.dep_dtml - flt1.arr_dtml).total_seconds() / 60.0
        return self.ml.min_connection_time_mins <= conn_mins <= self.ml.max_connection_time_mins

    def _check_total_time(
        self, pax: Passenger, flt_first: Flight, flt_last: Flight, max_mins: float
    ) -> bool:
        if flt_first.dep_dtml is None or flt_last.arr_dtml is None:
            return True
        total = (flt_last.arr_dtml - flt_first.dep_dtml).total_seconds() / 60.0
        return total <= max_mins

    def _get_max_travel_time(self, pax: Passenger) -> float:
        orig_dur = pax.original_duration_mins
        if orig_dur and orig_dur > 0:
            return min(
                orig_dur * self.ml.max_travel_time_multiplier,
                self.ml.max_total_travel_time_mins,
            )
        return self.ml.max_total_travel_time_mins

    def _get_candidate_cabins(self, pax: Passenger, flt: Flight) -> List[str]:
        """Return the set of cabins this passenger may use on this flight.

          - Y-only passengers may NOT be upgraded to C or F (hard constraint).
          - C/F passengers may use their own cabin OR downgrade to Y.
        """
        if self.pp.same_cabin_only:
            return [pax.cabin_cd]
        cabin = pax.cabin_cd.upper()
        if cabin == "Y":
            # Y-class: no upgrade permitted
            return ["Y"]
        # C or F: own cabin or downgrade to Y
        return [cabin, "Y"]

    def _check_capacity(self, flt: Flight, cabin: str) -> bool:
        """Return True only if this flight-cabin has genuine room for new pax.
        """
        avail = flt.available_seats(cabin)
        if avail >= self.pp.min_available_seats:
            return True
        if self.w.overbooking_allowed and self.w.overbooking_limit_fraction > 0:
            aul = flt.c_aul_cnt if cabin == "C" else flt.y_aul_cnt
            buffer = max(0, int(aul * self.w.overbooking_limit_fraction))
            if buffer == 0:
                return False
            current = flt.c_pax_cnt if cabin == "C" else flt.y_pax_cnt
            return current < aul + buffer
        return False

    def _top_k_itineraries(
        self, pax: Passenger, itineraries: List[Itinerary], k: int
    ) -> List[Itinerary]:
        scored = [(self._score_itinerary(pax, itin), itin) for itin in itineraries]
        scored.sort(key=lambda x: x[0])
        return [itin for _, itin in scored[:k]]

    def _score_itinerary(self, pax: Passenger, itin: Itinerary) -> float:
        """Heuristic score (lower = better) for ranking itineraries."""
        score = 0.0
        score += (itin.num_legs - 1) * self.ml.per_leg_penalty

        first_flt = self.flights[itin.legs[0][0]]
        if pax.dep_dtml and first_flt.dep_dtml:
            score += (
                abs((first_flt.dep_dtml - pax.dep_dtml).total_seconds()) / 60.0
            )

        last_flt = self.flights[itin.legs[-1][0]]
        if first_flt.dep_dtml and last_flt.arr_dtml:
            total_mins = (
                last_flt.arr_dtml - first_flt.dep_dtml
            ).total_seconds() / 60.0
            score += total_mins * 0.1

        for _, cabin in itin.legs:
            if cabin != pax.cabin_cd:
                score += 200.0 if (pax.cabin_cd == "C" and cabin == "Y") else 50.0

        for j, c in itin.legs:
            score -= min(self.flights[j].available_seats(c), 10) * 5

        return score




class PreprocessingEngine:
    """Preprocessing with itinerary-aware candidate generation.

    Selects passengers, builds their candidate itineraries, estimates QUBO
    size, picks a batching strategy, and returns a list of batch dicts ready
    for QUBOFormulator.
    """

    def __init__(
        self,
        processor: DataProcessor,
        config: PreprocessingConfig,
        weights: QUBOWeights,
    ):
        self.proc = processor
        self.cfg = config
        self.w = weights
        self.itinerary_builder: Optional[ItineraryBuilder] = None
        self.stats: dict = {}

    def prepare(self) -> List[dict]:
        passengers = self._select_passengers()

        self.itinerary_builder = ItineraryBuilder(
            flights=self.proc.available_flights,
            ml_config=self.cfg.multi_leg,
            pp_config=self.cfg,
            weights=self.w,
        )
        itineraries = self.itinerary_builder.build_itineraries(passengers)

        self.stats.update(self.itinerary_builder.stats)
        self.stats["total_passengers"] = len(passengers)
        self.stats["total_itineraries"] = sum(len(v) for v in itineraries.values())

        estimated_vars = self._estimate_variables(passengers, itineraries)
        self.stats["estimated_total_vars"] = estimated_vars
        logger.info("Estimated QUBO variables: %d", estimated_vars)

        strategy = self._resolve_strategy(passengers, estimated_vars)

        # Pass already-built itineraries into _create_batches so _partition
        # can sample from them
        batches = self._create_batches(passengers, itineraries, strategy)

        self.stats["num_batches"] = len(batches)
        self.stats["batch_strategy"] = strategy.value
        for i, b in enumerate(batches):
            logger.info(
                "  Batch %d [%s]: %d pax, %d itins",
                i,
                b["batch_id"],
                len(b["passengers"]),
                sum(len(v) for v in b["itineraries"].values()),
            )
        return batches

    def _select_passengers(self) -> List[Passenger]:
        passengers = list(self.proc.affected_passengers)
        if self.cfg.include_non_affected_passengers:
            logger.warning("Including non-affected passengers!")
            affected_routes = {p.route_key for p in passengers}
            non_aff = []
            for p in self.proc.non_affected_passengers:
                if self.cfg.non_affected_same_route_only:
                    if p.route_key not in affected_routes:
                        continue
                non_aff.append(p)
            non_aff.sort(key=lambda p: p.cvm)
            passengers.extend(non_aff[: self.cfg.max_non_affected_passengers])
        return passengers

    def _estimate_variables(self, passengers, itineraries) -> int:
        return sum(len(v) for v in itineraries.values()) + len(passengers)

    def _resolve_strategy(self, passengers, est_vars) -> BatchStrategy:
        if self.cfg.batch_strategy != BatchStrategy.AUTO:
            return self.cfg.batch_strategy
        if est_vars <= self.cfg.max_qubo_variables:
            return BatchStrategy.NONE
        routes = {p.route_key for p in passengers}
        if len(routes) > 1:
            return BatchStrategy.BY_ROUTE_AND_TIME
        return BatchStrategy.BY_TIME_WINDOW

    def _create_batches(self, passengers, itineraries, strategy):
        if strategy == BatchStrategy.NONE:
            return [
                self._make_batch(
                    passengers, itineraries, list(range(len(passengers))), "batch_0"
                )
            ]
        partition = self._partition(passengers, itineraries, strategy)
        return [
            self._make_batch(passengers, itineraries, indices, label)
            for label, indices in partition.items()
        ]

    def _partition(
        self, passengers, itineraries, strategy
    ) -> Dict[str, List[int]]:
        """Partition passengers into batch groups."""

        groups: Dict[str, List[int]] = defaultdict(list)

        if strategy == BatchStrategy.BY_ROUTE:
            for i, p in enumerate(passengers):
                groups[f"rt_{p.route_key}"].append(i)

        elif strategy == BatchStrategy.BY_TIME_WINDOW:
            w = self.cfg.time_batch_window_hours
            times = [(i, p.dep_dtml) for i, p in enumerate(passengers) if p.dep_dtml]
            no_time = [i for i, p in enumerate(passengers) if not p.dep_dtml]
            if times:
                min_t = min(t for _, t in times)
                for i, dt in times:
                    block = int((dt - min_t).total_seconds() / 3600.0 / w)
                    groups[f"time_{block}"].append(i)
            if no_time:
                first_key = next(iter(groups)) if groups else "time_0"
                groups[first_key].extend(no_time)

        elif strategy == BatchStrategy.BY_ROUTE_AND_TIME:
            route_groups: Dict[str, List[int]] = defaultdict(list)
            for i, p in enumerate(passengers):
                route_groups[p.route_key].append(i)

            for route, indices in route_groups.items():
                # Estimate using already-built itinerary counts
                sample = indices[:5]
                sample_itins = sum(len(itineraries.get(i, [])) for i in sample)
                est = (sample_itins * len(indices) // max(len(sample), 1))

                if est <= self.cfg.target_batch_variables:
                    groups[f"rt_{route}"] = indices
                else:
                    w = self.cfg.time_batch_window_hours
                    times = [
                        (i, passengers[i].dep_dtml)
                        for i in indices
                        if passengers[i].dep_dtml
                    ]
                    if times:
                        min_t = min(t for _, t in times)
                        for i, dt in times:
                            block = int(
                                (dt - min_t).total_seconds() / 3600 / w
                            )
                            groups[f"rt_{route}_t{block}"].append(i)
                    else:
                        groups[f"rt_{route}"] = indices

        elif strategy == BatchStrategy.BY_PRIORITY_TIER:
            cvm_vals = np.array([p.cvm for p in passengers], dtype=float)

            # Determine interior cut points (ascending CVM boundaries)
            if self.cfg.priority_tiers:
                # if user supplies explicit split values.
                # e.g. [2.0, 5.0, 9.0] → 4 bins: ≤2, (2,5], (5,9], >9
                cut_points = np.sort(np.unique(
                    np.asarray(self.cfg.priority_tiers, dtype=float)
                ))
            else:
                # Based on number of bins property:
                # equal-population quantile bins from actual CVM data.
                n = max(2, self.cfg.priority_bins)
                pcts = np.linspace(0, 100, n + 1)[1:-1]
                cut_points = np.unique(np.percentile(cvm_vals, pcts))

            n_bins = len(cut_points) + 1  # always ≥ 1

            raw_bin = np.clip(
                np.digitize(cvm_vals, cut_points, right=True), 0, n_bins - 1
            )

            # Invert: priority rank 0 → highest CVM → runner processes it first
            priority_rank = (n_bins - 1) - raw_bin

            # Zero-padded labels so alphabetic order == priority order for any n
            width = len(str(n_bins))
            for i, rank in enumerate(priority_rank):
                label = f"priority_{int(rank) + 1:0{width}d}_of_{n_bins:0{width}d}"
                groups[label].append(i)

            # Explicit sort so highest CVM is first
            groups = dict(sorted(groups.items()))

            # Build boundary arrays for logging and stats
            lo_vals = np.concatenate([[cvm_vals.min()], cut_points])
            hi_vals = np.concatenate([cut_points, [cvm_vals.max()]])
            logger.info("Priority tier CVM boundaries (%d bins):", n_bins)
            tier_info = []
            for rank in range(n_bins):
                b = (n_bins - 1) - rank  # raw_bin index for this rank
                cnt = int((raw_bin == b).sum())
                lbl = f"priority_{rank + 1:0{width}d}_of_{n_bins:0{width}d}"
                logger.info(
                    "  %s  CVM [%.4f, %.4f]  %d pax", lbl, lo_vals[b], hi_vals[b], cnt
                )
                tier_info.append({
                    "label": lbl,
                    "cvm_lo": float(lo_vals[b]),
                    "cvm_hi": float(hi_vals[b]),
                    "pax_count": cnt,
                })
            # stats for potential debugging
            self.stats["priority_tier_info"] = tier_info
            self.stats["priority_tier_cuts"] = cut_points.tolist()

        elif strategy == BatchStrategy.BY_CABIN:
            for i, p in enumerate(passengers):
                groups[f"cabin_{p.cabin_cd}"].append(i)

        if not groups:
            groups["all"] = list(range(len(passengers)))

        return groups

    def _make_batch(self, all_pax, all_itins, indices, batch_id):
        pax_list = [all_pax[i] for i in indices]
        new_itins = {
            new_i: all_itins.get(old_i, []) for new_i, old_i in enumerate(indices)
        }
        batch_reclocs = {p.recloc for p in pax_list}
        conn = {
            gid: [p for p in grp if p.recloc in batch_reclocs]
            for gid, grp in self.proc.connection_groups.items()
        }
        conn = {k: v for k, v in conn.items() if len(v) > 1}
        return {
            "passengers": pax_list,
            "flights": self.proc.available_flights,
            "itineraries": new_itins,
            "connection_groups": conn,
            "batch_id": batch_id,
            "original_indices": list(indices),
        }

    def get_stats(self) -> dict:
        return self.stats.copy()
