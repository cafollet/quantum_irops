"""QUBO formulation and solver backends.

QUBOFormulator builds the objective matrix from passengers, flights, and
candidate itineraries.  QUBOSolver provides SA, neal, and D-Wave backends.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from .config import PreprocessingConfig, QUBOWeights
from .models import Flight, Passenger
from .types import Itinerary

logger = logging.getLogger(__name__)



class QUBOFormulator:
    """Builds QUBO with itinerary-level decision variables.

    Decision variables:
        x_{i,k} = 1 if passenger i is assigned itinerary k
        s_i      = 1 if passenger i is unbooked (slack)

    Each itinerary k encodes the full path (possibly multi-leg)
    including cabin choices per leg.
    """

    def __init__(
        self,
        passengers: List[Passenger],
        flights: List[Flight],
        itineraries: Dict[int, List[Itinerary]],
        connection_groups: Dict[str, List[Passenger]],
        weights: QUBOWeights,
        pp_config: PreprocessingConfig,
    ):
        self.passengers = passengers
        self.flights = flights
        self.itineraries = itineraries
        self.conn_groups = connection_groups
        self.w = weights
        self.pp = pp_config

        self.var_map: Dict[Tuple, int] = {}
        self.index_to_var: Dict[int, Tuple] = {}
        self.n_vars = 0
        self.Q: Dict[Tuple[int, int], float] = {}

    def build(self) -> Dict[Tuple[int, int], float]:
        logger.info(
            "Building QUBO: %d pax, %d itineraries",
            len(self.passengers),
            sum(len(v) for v in self.itineraries.values()),
        )
        self._create_variables()
        self._add_assignment_constraint()
        self._add_capacity_constraint()
        self._add_connection_constraint()
        self._add_unbooked_penalty()
        self._add_time_change_penalty()
        self._add_cabin_stability_penalty()
        self._add_multi_leg_penalty()
        self._add_priority_reward()
        self._add_time_window_soft_penalty()
        self._add_non_affected_penalty()
        self._scale()
        logger.info("QUBO: %d vars, %d non-zero entries", self.n_vars, len(self.Q))
        return self.Q

    def _create_variables(self):
        idx = 0
        for i in range(len(self.passengers)):
            for k, itin in enumerate(self.itineraries.get(i, [])):
                self.var_map[(i, k)] = idx
                self.index_to_var[idx] = ("assign", i, k)
                idx += 1
        for i in range(len(self.passengers)):
            self.var_map[("slack", i)] = idx
            self.index_to_var[idx] = ("slack", i)
            idx += 1
        self.n_vars = idx

    def _add_to_Q(self, i: int, j: int, val: float):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        self.Q[(i, j)] = self.Q.get((i, j), 0.0) + val

    # CONSTRAINT: one assignment per passenger

    def _add_assignment_constraint(self):
        P = self.w.one_assignment_penalty
        for i in range(len(self.passengers)):
            vars_i = [
                self.var_map[(i, k)]
                for k in range(len(self.itineraries.get(i, [])))
            ]
            vars_i.append(self.var_map[("slack", i)])
            for v in vars_i:
                self._add_to_Q(v, v, -P)
            for a in range(len(vars_i)):
                for b in range(a + 1, len(vars_i)):
                    self._add_to_Q(vars_i[a], vars_i[b], 2.0 * P)

    # CONSTRAINT: capacity across all legs

    def _add_capacity_constraint(self):
        """For each (flight, cabin), the sum of assigned pax_cnt must not
        exceed the available seats (or soft-cap when overbooking is allowed).

        Correct QUBO penalty for the inequality constraint
            sum_i (cnt_i * x_i) <= C
        is the quadratic penalty term
            P * (sum_i cnt_i * x_i  -  C)^2

        Expanding (x_i binary, so x_i^2 = x_i):
            diagonal :  P * cnt_i^2 - 2*P*C*cnt_i   per variable
            cross     :  2 * P * cnt_i * cnt_j        per distinct pair


        When overbooking is disabled C = avail (remaining available seats).
        When overbooking is enabled   C = soft_cap - already_booked.
        """
        P = self.w.capacity_violation_penalty

        # Collect all (variable, pax_cnt) tuples that share a (flight, cabin).
        fc_vars: Dict[Tuple[int, str], List[Tuple[int, int, int]]] = defaultdict(list)
        for i in range(len(self.passengers)):
            for k, itin in enumerate(self.itineraries.get(i, [])):
                var_idx = self.var_map[(i, k)]
                cnt = self.passengers[i].pax_cnt
                for j, c in itin.legs:
                    fc_vars[(j, c)].append((i, var_idx, cnt))

        for (j, c), entries in fc_vars.items():
            flt = self.flights[j]

            # C = maximum NEW passengers that may land on this (flight, cabin).
            if self.w.overbooking_allowed:
                pax_booked = flt.c_pax_cnt if c == "C" else flt.y_pax_cnt
                cap_limit = flt.max_capacity(c, self.w.overbooking_limit_fraction)
                C = max(0, cap_limit - pax_booked)
            else:
                C = flt.available_seats(c)

            # diagonal terms: P*cnt_i^2 - 2*P*C*cnt_i
            for _pi, vi, cnt in entries:
                self._add_to_Q(vi, vi, P * cnt * cnt - 2.0 * P * C * cnt)

            # off-diagonal terms: 2*P*cnt_i*cnt_j (all distinct pax pairs)
            for a in range(len(entries)):
                for b in range(a + 1, len(entries)):
                    pi_a, vi_a, cnt_a = entries[a]
                    pi_b, vi_b, cnt_b = entries[b]
                    # Skip same-passenger pairs: the one-assignment constraint
                    # already prevents both from being selected simultaneously.
                    if pi_a == pi_b:
                        continue
                    self._add_to_Q(vi_a, vi_b, 2.0 * P * cnt_a * cnt_b)

    # CONSTRAINT: connections

    def _add_connection_constraint(self):
        P = self.w.connection_break_penalty
        pax_map = {id(p): i for i, p in enumerate(self.passengers)}

        for gid, grp in self.conn_groups.items():
            sorted_segs = sorted(
                grp, key=lambda p: p.dep_dtml if p.dep_dtml else datetime.min
            )
            for s in range(len(sorted_segs) - 1):
                ci = pax_map.get(id(sorted_segs[s]))
                ni = pax_map.get(id(sorted_segs[s + 1]))
                if ci is None or ni is None:
                    continue
                for k1, itin1 in enumerate(self.itineraries.get(ci, [])):
                    for k2, itin2 in enumerate(self.itineraries.get(ni, [])):
                        last_flt1 = self.flights[itin1.legs[-1][0]]
                        first_flt2 = self.flights[itin2.legs[0][0]]
                        if not self._valid_itinerary_connection(
                            last_flt1, first_flt2, sorted_segs[s].conn_time_mins
                        ):
                            v1 = self.var_map[(ci, k1)]
                            v2 = self.var_map[(ni, k2)]
                            self._add_to_Q(v1, v2, P)

    @staticmethod
    def _valid_itinerary_connection(flt1, flt2, min_conn):
        if flt1.arr_dtml is None or flt2.dep_dtml is None:
            return False
        if flt1.dest_cd != flt2.orig_cd:
            return False
        conn = (flt2.dep_dtml - flt1.arr_dtml).total_seconds() / 60
        return max(min_conn, 30) <= conn <= 23 * 60

    # OBJECTIVE: unbooked penalty

    def _add_unbooked_penalty(self):
        """Penalise leaving an affected passenger unbooked.
        """
        for i, pax in enumerate(self.passengers):
            si = self.var_map[("slack", i)]
            if pax.is_affected:
                self._add_to_Q(
                    si,
                    si,
                    self.w.unbooked_passenger_penalty * pax.pax_cnt,
                )

    # OBJECTIVE: time change

    def _add_time_change_penalty(self):
        for i, pax in enumerate(self.passengers):
            for k, itin in enumerate(self.itineraries.get(i, [])):
                vi = self.var_map[(i, k)]
                first_flt = self.flights[itin.legs[0][0]]
                last_flt = self.flights[itin.legs[-1][0]]
                dep_d = self._td(pax.dep_dtml, first_flt.dep_dtml)
                arr_d = self._td(pax.arr_dtml, last_flt.arr_dtml)
                penalty = (
                    self.w.departure_time_change_weight * dep_d
                    + self.w.arrival_time_change_weight * arr_d
                )
                self._add_to_Q(vi, vi, penalty)

    @staticmethod
    def _td(dt1, dt2):
        if dt1 and dt2:
            return abs((dt2 - dt1).total_seconds()) / 60.0
        return 0.0

    # OBJECTIVE: cabin stability

    def _add_cabin_stability_penalty(self):
        for i, pax in enumerate(self.passengers):
            for k, itin in enumerate(self.itineraries.get(i, [])):
                vi = self.var_map[(i, k)]
                for j, cabin in itin.legs:
                    if pax.cabin_cd == "C" and cabin == "Y":
                        self._add_to_Q(
                            vi, vi, self.w.cabin_downgrade_penalty * pax.pax_cnt
                        )
                    elif pax.cabin_cd == "Y" and cabin == "C":
                        self._add_to_Q(
                            vi, vi, self.w.cabin_upgrade_penalty * pax.pax_cnt
                        )

    # OBJECTIVE: multi-leg penalty

    def _add_multi_leg_penalty(self):
        """Penalize itineraries with more legs (prefer direct)."""
        if not self.pp.multi_leg.enable_multi_leg:
            return
        for i in range(len(self.passengers)):
            for k, itin in enumerate(self.itineraries.get(i, [])):
                if itin.num_legs > 1:
                    vi = self.var_map[(i, k)]
                    extra_legs = itin.num_legs - 1
                    penalty = self.w.multi_leg_per_leg_penalty * extra_legs
                    first_flt = self.flights[itin.legs[0][0]]
                    last_flt = self.flights[itin.legs[-1][0]]
                    if first_flt.dep_dtml and last_flt.arr_dtml:
                        total_mins = (
                            last_flt.arr_dtml - first_flt.dep_dtml
                        ).total_seconds() / 60.0
                        penalty += (
                            self.w.multi_leg_total_time_penalty_weight * total_mins
                        )
                    self._add_to_Q(vi, vi, penalty)

    # OBJECTIVE: priority reward

    def _add_priority_reward(self):
        for i, pax in enumerate(self.passengers):
            if not pax.is_affected:
                continue
            priority = (
                self.w.cvm_priority_weight * pax.cvm
                + self.w.group_size_weight * pax.pax_cnt
            )
            for k in range(len(self.itineraries.get(i, []))):
                vi = self.var_map[(i, k)]
                self._add_to_Q(vi, vi, -priority)

    # SOFT: time window
    def _add_time_window_soft_penalty(self):
        window_mins = self.w.time_window_hours * 60
        for i, pax in enumerate(self.passengers):
            for k, itin in enumerate(self.itineraries.get(i, [])):
                first_flt = self.flights[itin.legs[0][0]]
                dep_d = self._td(pax.dep_dtml, first_flt.dep_dtml)
                if dep_d > window_mins:
                    vi = self.var_map[(i, k)]
                    self._add_to_Q(
                        vi, vi, self.w.time_window_soft_penalty * (dep_d - window_mins)
                    )

    # Non-affected penalty

    def _add_non_affected_penalty(self):
        if not self.pp.include_non_affected_passengers:
            return
        for i, pax in enumerate(self.passengers):
            if pax.is_affected:
                continue
            for k in range(len(self.itineraries.get(i, []))):
                vi = self.var_map[(i, k)]
                self._add_to_Q(vi, vi, self.pp.non_affected_move_penalty * pax.pax_cnt)

    def _scale(self):
        if self.w.global_scale != 1.0:
            for key in self.Q:
                self.Q[key] *= self.w.global_scale

    def get_variable_map(self):
        return self.var_map.copy()

    def get_reverse_map(self):
        return self.index_to_var.copy()


class QUBOSolver:
    @staticmethod
    def solve_simulated_annealing(
        Q,
        n_vars,
        num_reads=100,
        T_init=100.0,
        T_min=0.01,
        alpha=0.995,
        seed=42,
    ):
        rng = np.random.RandomState(seed)
        best_sol = None
        best_e = float("inf")

        for _ in range(num_reads):
            state = rng.randint(0, 2, size=n_vars)
            energy = QUBOSolver._energy(Q, state)
            T = T_init
            while T > T_min:
                flip = rng.randint(0, n_vars)
                delta = QUBOSolver._flip_delta(Q, state, flip, n_vars)
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    state[flip] = 1 - state[flip]
                    energy += delta
                T *= alpha
            if energy < best_e:
                best_e = energy
                best_sol = state.copy()

        logger.info("SA best energy: %.2f", best_e)
        return {i: int(best_sol[i]) for i in range(n_vars)}

    @staticmethod
    def _energy(Q, s):
        return sum(v * s[i] * s[j] for (i, j), v in Q.items())

    @staticmethod
    def _flip_delta(Q, state, flip, n_vars):
        nv = 1 - state[flip]
        d = 0.0
        k = (flip, flip)
        if k in Q:
            d += Q[k] * (nv - state[flip])
        for o in range(n_vars):
            if o == flip:
                continue
            k = (min(flip, o), max(flip, o))
            if k in Q:
                d += Q[k] * state[o] * (nv - state[flip])
        return d

    @staticmethod
    def solve_neal(Q, num_reads=1000, **kw):
        import neal

        sampler = neal.SimulatedAnnealingSampler()
        resp = sampler.sample_qubo(Q, num_reads=num_reads, **kw)
        logger.info("Neal energy: %.2f", resp.first.energy)
        return resp.first.sample

    @staticmethod
    def solve_dwave(Q, num_reads=1000, **kw):
        from dwave.system import DWaveSampler, EmbeddingComposite

        sampler = EmbeddingComposite(DWaveSampler())
        resp = sampler.sample_qubo(Q, num_reads=num_reads, **kw)
        logger.info("D-Wave energy: %.2f", resp.first.energy)
        return resp.first.sample
