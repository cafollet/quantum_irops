"""Post-solve result handling.

SolutionInterpreter translates QUBO bit-assignments back into the output
DataFrame schema.  CapacityTracker maintains remaining seat counts across
batches so later batches see accurate availability.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

from .models import Flight, Passenger
from .types import Itinerary

logger = logging.getLogger(__name__)

# Output column order
OUTPUT_COLUMNS = [
    "RECLOC",
    "CABIN_CD",
    "COS_CD",
    "OPER_OD_ORIG_CD",
    "OPER_OD_DEST_CD",
    "DEP_KEY",
    "DEP_DT",
    "ORIG_CD",
    "DEST_CD",
    "FLT_NUM",
    "DEP_DTML",
    "ARR_DTML",
    "DEP_DTMZ",
    "ARR_DTMZ",
    "PREV_OD_BROKEN_IND",
    "PAX_CNT",
    "CVM",
    "PREV_CONN_TIME_MINS",
    "ALT_DEP_KEY",
    "ALT_CABIN_CD",
    "ALT_OPER_OD_ORIG_CD",
    "ALT_OPER_OD_DEST_CD",
    "ALT_DEP_DT",
    "ALT_ORIG_CD",
    "ALT_DEST_CD",
    "ALT_FLT_NUM",
    "ALT_DEP_DTML",
    "ALT_ARR_DTML",
    "ALT_DEP_DTMZ",
    "ALT_ARR_DTMZ",
    "ALT_CONN_TIME_MINS",
    "IS_AFFECTED",
    "IS_DIRECT",
    "ORIG_CABIN",
    "ITINERARY_CABINS",
    "DEP_CHANGE_MINS",
]


def _fmt(dt, fmt):
    return dt.strftime(fmt) if dt else ""


class SolutionInterpreter:
    """Interprets itinerary-level solution into the required output schema."""

    def __init__(self, passengers, flights, itineraries, index_to_var):
        self.passengers: List[Passenger] = passengers
        self.flights: List[Flight] = flights
        self.itineraries: Dict[int, List[Itinerary]] = itineraries
        self.index_to_var: Dict[int, Tuple] = index_to_var

    def interpret(self, solution) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assignments = []
        unbooked = []

        # Guard against solver constraint violations: the one-assignment
        # penalty may not be tight enough for the SA to converge, leaving
        # multiple x_{i,k}=1 for the same passenger segment.
        # Key is passenger index i — unique per (RECLOC, DEP_KEY) segment
        # within this batch.
        assigned_pax: set = set()
        unbooked_pax: set = set()  # tracks passengers added to unbooked list

        for idx, val in solution.items():
            if val != 1:
                continue
            info = self.index_to_var.get(idx)
            if info is None:
                continue

            if info[0] == "assign":
                _, i, k = info

                if i in assigned_pax:
                    logger.debug(
                        "Passenger %d already assigned (idx=%d k=%d); "
                        "skipping extra (solver constraint violation).",
                        i, idx, k,
                    )
                    continue

                assigned_pax.add(i)
                pax = self.passengers[i]
                itin = self.itineraries[i][k]

                # _itin_id groups all alternate legs of this single assignment
                # together and is used by _merge_results for cross-batch
                # deduplication.  Including pax.dep_key (the ORIGINAL flight
                # key) means two segments of the same RECLOC (e.g. outbound
                # leg DEP_KEY_A and return leg DEP_KEY_B) produce distinct
                # _itin_ids and are never collapsed into one.
                first_alt_key = self.flights[itin.legs[0][0]].dep_key
                itin_id = f"{pax.recloc}|{pax.dep_key}|{first_alt_key}"

                for leg_num, (j, cabin) in enumerate(itin.legs):
                    flt = self.flights[j]

                    alt_conn_time = 0.0
                    if leg_num < len(itin.legs) - 1:
                        next_j, _ = itin.legs[leg_num + 1]
                        next_flt = self.flights[next_j]
                        if flt.arr_dtml and next_flt.dep_dtml:
                            alt_conn_time = (
                                next_flt.dep_dtml - flt.arr_dtml
                            ).total_seconds() / 60.0

                    first_flt = self.flights[itin.legs[0][0]]
                    last_flt = self.flights[itin.legs[-1][0]]

                    assignments.append(
                        {
                            "RECLOC": pax.recloc,
                            "CABIN_CD": pax.cabin_cd,
                            "COS_CD": pax.cos_cd,
                            "OPER_OD_ORIG_CD": pax.oper_od_orig_cd,
                            "OPER_OD_DEST_CD": pax.oper_od_dest_cd,
                            "DEP_KEY": pax.dep_key,
                            "DEP_DT": _fmt(pax.dep_dtml, "%m/%d/%Y"),
                            "ORIG_CD": pax.orig_cd,
                            "DEST_CD": pax.dest_cd,
                            "FLT_NUM": pax.flt_num,
                            "DEP_DTML": _fmt(pax.dep_dtml, "%m/%d/%Y %H:%M"),
                            "ARR_DTML": _fmt(pax.arr_dtml, "%m/%d/%Y %H:%M"),
                            "DEP_DTMZ": _fmt(pax.dep_dtmz, "%m/%d/%Y %H:%M"),
                            "ARR_DTMZ": _fmt(pax.arr_dtmz, "%m/%d/%Y %H:%M"),
                            "PREV_OD_BROKEN_IND": pax.od_broken_ind,
                            "PAX_CNT": pax.pax_cnt,
                            "CVM": pax.cvm,
                            "PREV_CONN_TIME_MINS": pax.conn_time_mins,
                            "ALT_DEP_KEY": flt.dep_key,
                            "ALT_CABIN_CD": cabin,
                            "ALT_OPER_OD_ORIG_CD": first_flt.orig_cd,
                            "ALT_OPER_OD_DEST_CD": last_flt.dest_cd,
                            "ALT_DEP_DT": _fmt(flt.dep_dtml, "%m/%d/%Y"),
                            "ALT_ORIG_CD": flt.orig_cd,
                            "ALT_DEST_CD": flt.dest_cd,
                            "ALT_FLT_NUM": flt.flt_num,
                            "ALT_DEP_DTML": _fmt(flt.dep_dtml, "%m/%d/%Y %H:%M"),
                            "ALT_ARR_DTML": _fmt(flt.arr_dtml, "%m/%d/%Y %H:%M"),
                            "ALT_DEP_DTMZ": _fmt(flt.dep_dtmz, "%m/%d/%Y %H:%M"),
                            "ALT_ARR_DTMZ": _fmt(flt.arr_dtmz, "%m/%d/%Y %H:%M"),
                            "ALT_CONN_TIME_MINS": alt_conn_time,
                            "IS_AFFECTED": pax.is_affected,
                            "IS_DIRECT": itin.is_direct,
                            "ORIG_CABIN": pax.cabin_cd,
                            "ITINERARY_CABINS": "/".join(itin.cabins),
                            "DEP_CHANGE_MINS": (
                                abs(
                                    (
                                        first_flt.dep_dtml - pax.dep_dtml
                                    ).total_seconds()
                                    / 60.0
                                )
                                if first_flt.dep_dtml and pax.dep_dtml
                                else 0.0
                            ),
                            # Internal tag — stripped by _merge_results after
                            # cross-batch deduplication.
                            "_itin_id": itin_id,
                        }
                    )

            elif info[0] == "slack":
                _, i = info
                # If the passenger segment was already assigned, don't also
                # emit an unbooked row (another form of constraint violation).
                if i in assigned_pax:
                    continue
                pax = self.passengers[i]
                if pax.is_affected:
                    unbooked_pax.add(i)
                    unbooked.append(
                        {
                            "RECLOC": pax.recloc,
                            "CABIN_CD": pax.cabin_cd,
                            "COS_CD": pax.cos_cd,
                            "OPER_OD_ORIG_CD": pax.oper_od_orig_cd,
                            "OPER_OD_DEST_CD": pax.oper_od_dest_cd,
                            "DEP_KEY": pax.dep_key,
                            "DEP_DT": _fmt(pax.dep_dtml, "%m/%d/%Y"),
                            "ORIG_CD": pax.orig_cd,
                            "DEST_CD": pax.dest_cd,
                            "FLT_NUM": pax.flt_num,
                            "DEP_DTML": _fmt(pax.dep_dtml, "%m/%d/%Y %H:%M"),
                            "ARR_DTML": _fmt(pax.arr_dtml, "%m/%d/%Y %H:%M"),
                            "DEP_DTMZ": _fmt(pax.dep_dtmz, "%m/%d/%Y %H:%M"),
                            "ARR_DTMZ": _fmt(pax.arr_dtmz, "%m/%d/%Y %H:%M"),
                            "PREV_OD_BROKEN_IND": pax.od_broken_ind,
                            "PAX_CNT": pax.pax_cnt,
                            "CVM": pax.cvm,
                            "PREV_CONN_TIME_MINS": pax.conn_time_mins,
                            "ALT_DEP_KEY": "",
                            "ALT_CABIN_CD": "",
                            "ALT_OPER_OD_ORIG_CD": "",
                            "ALT_OPER_OD_DEST_CD": "",
                            "ALT_DEP_DT": "",
                            "ALT_ORIG_CD": "",
                            "ALT_DEST_CD": "",
                            "ALT_FLT_NUM": "",
                            "ALT_DEP_DTML": "",
                            "ALT_ARR_DTML": "",
                            "ALT_DEP_DTMZ": "",
                            "ALT_ARR_DTMZ": "",
                            "ALT_CONN_TIME_MINS": "",
                            "IS_AFFECTED": True,
                            "IS_DIRECT": False,
                            "ORIG_CABIN": pax.cabin_cd,
                            "ITINERARY_CABINS": "",
                            "DEP_CHANGE_MINS": 0.0,
                        }
                    )

        # Force any affected passenger that the solver
        # left in the all-zeros state (neither assigned nor slack=1) into unbooked.
        # This happens when the all-zeros QUBO energy is lower than both the slack and assignment diagonals
        for i, pax in enumerate(self.passengers):
            if pax.is_affected and i not in assigned_pax and i not in unbooked_pax:
                logger.warning(
                    "Passenger %d (%s|%s) had all variables=0 in solver output "
                    "(constraint violation / weight imbalance); forcing unbooked.",
                    i, pax.recloc, pax.dep_key,
                )
                unbooked.append(
                    {
                        "RECLOC": pax.recloc,
                        "CABIN_CD": pax.cabin_cd,
                        "COS_CD": pax.cos_cd,
                        "OPER_OD_ORIG_CD": pax.oper_od_orig_cd,
                        "OPER_OD_DEST_CD": pax.oper_od_dest_cd,
                        "DEP_KEY": pax.dep_key,
                        "DEP_DT": _fmt(pax.dep_dtml, "%m/%d/%Y"),
                        "ORIG_CD": pax.orig_cd,
                        "DEST_CD": pax.dest_cd,
                        "FLT_NUM": pax.flt_num,
                        "DEP_DTML": _fmt(pax.dep_dtml, "%m/%d/%Y %H:%M"),
                        "ARR_DTML": _fmt(pax.arr_dtml, "%m/%d/%Y %H:%M"),
                        "DEP_DTMZ": _fmt(pax.dep_dtmz, "%m/%d/%Y %H:%M"),
                        "ARR_DTMZ": _fmt(pax.arr_dtmz, "%m/%d/%Y %H:%M"),
                        "PREV_OD_BROKEN_IND": pax.od_broken_ind,
                        "PAX_CNT": pax.pax_cnt,
                        "CVM": pax.cvm,
                        "PREV_CONN_TIME_MINS": pax.conn_time_mins,
                        "ALT_DEP_KEY": "",
                        "ALT_CABIN_CD": "",
                        "ALT_OPER_OD_ORIG_CD": "",
                        "ALT_OPER_OD_DEST_CD": "",
                        "ALT_DEP_DT": "",
                        "ALT_ORIG_CD": "",
                        "ALT_DEST_CD": "",
                        "ALT_FLT_NUM": "",
                        "ALT_DEP_DTML": "",
                        "ALT_ARR_DTML": "",
                        "ALT_DEP_DTMZ": "",
                        "ALT_ARR_DTMZ": "",
                        "ALT_CONN_TIME_MINS": "",
                        "IS_AFFECTED": True,
                        "IS_DIRECT": False,
                        "ORIG_CABIN": pax.cabin_cd,
                        "ITINERARY_CABINS": "",
                        "DEP_CHANGE_MINS": 0.0,
                    }
                )

        # Build Dfs. Keep _itin_id so _merge_results can deduplicate
        # cross-batch duplicates without discarding multi-leg alt legs.
        if assignments:
            assign_df = pd.DataFrame(assignments)
            cols = [c for c in OUTPUT_COLUMNS if c in assign_df.columns] + ["_itin_id"]
            assign_df = assign_df[cols]
        else:
            assign_df = pd.DataFrame(columns=OUTPUT_COLUMNS + ["_itin_id"])

        unbook_df = pd.DataFrame(unbooked, columns=OUTPUT_COLUMNS)
        return assign_df, unbook_df


class CapacityTracker:
    """Tracks remaining seat counts across batches."""

    def __init__(self, flights: List[Flight]):
        self.remaining_c: Dict[int, int] = {j: f.c_avail_cnt for j, f in enumerate(flights)}
        self.remaining_y: Dict[int, int] = {j: f.y_avail_cnt for j, f in enumerate(flights)}
        self.consumed_c: Dict[int, int] = {j: 0 for j in range(len(flights))}
        self.consumed_y: Dict[int, int] = {j: 0 for j in range(len(flights))}

    def consume(self, flight_idx: int, cabin: str, count: int):
        """Deduct count seats. Negative remaining = overbooked."""
        if cabin == "C":
            self.remaining_c[flight_idx] -= count
            self.consumed_c[flight_idx] += count
        else:
            self.remaining_y[flight_idx] -= count
            self.consumed_y[flight_idx] += count

    def overbooked_flights(self) -> List[Tuple[int, str, int]]:
        """Return list of (flight_idx, cabin, seats_over) for any overbooked cabin."""
        result = []
        for j, rem in self.remaining_c.items():
            if rem < 0:
                result.append((j, "C", -rem))
        for j, rem in self.remaining_y.items():
            if rem < 0:
                result.append((j, "Y", -rem))
        return result

    def update_flights(self, flights: List[Flight]) -> List[Flight]:
        """Return a new Flight list with availability adjusted for consumed seats."""
        updated = []
        for j, f in enumerate(flights):
            new_c_avail = max(0, self.remaining_c.get(j, 0))
            new_y_avail = max(0, self.remaining_y.get(j, 0))
            new_c_pax = f.c_pax_cnt + self.consumed_c.get(j, 0)
            new_y_pax = f.y_pax_cnt + self.consumed_y.get(j, 0)
            updated.append(
                Flight(
                    dep_key=f.dep_key,
                    dep_dt=f.dep_dt,
                    orig_cd=f.orig_cd,
                    dest_cd=f.dest_cd,
                    flt_num=f.flt_num,
                    dep_dtml=f.dep_dtml,
                    arr_dtml=f.arr_dtml,
                    dep_dtmz=f.dep_dtmz,
                    arr_dtmz=f.arr_dtmz,
                    c_cap_cnt=f.c_cap_cnt,
                    c_aul_cnt=f.c_aul_cnt,
                    c_pax_cnt=new_c_pax,
                    c_avail_cnt=new_c_avail,
                    y_cap_cnt=f.y_cap_cnt,
                    y_aul_cnt=f.y_aul_cnt,
                    y_pax_cnt=new_y_pax,
                    y_avail_cnt=new_y_avail,
                    original_index=f.original_index,
                )
            )
        return updated
