"""Data loading and ingestion layer.

Converts raw CSV/DataFrame input into Passenger and Flight domain objects,
identifies affected passengers, and builds connection groups.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .models import Flight, Passenger

logger = logging.getLogger(__name__)


class DataProcessor:
    DATETIME_FORMATS = [
        "%Y-%m-%d %H:%M:%S",  # ISO with seconds (actual CSV format)
        "%Y-%m-%d %H:%M",  # ISO without seconds
        "%Y-%m-%d",  # ISO date only
        "%m/%d/%Y %H:%M",  # US with time
        "%m/%d/%Y",  # US date only
    ]

    def __init__(self):
        self.all_passengers: List[Passenger] = []
        self.affected_passengers: List[Passenger] = []
        self.non_affected_passengers: List[Passenger] = []
        self.cancelled_flights: Dict[str, dict] = {}
        self.available_flights: List[Flight] = []
        self.connection_groups: Dict[str, List[Passenger]] = {}

    @staticmethod
    def _parse_datetime(val) -> Optional[datetime]:
        if pd.isna(val):
            return None
        if isinstance(val, datetime):
            return val
        for fmt in DataProcessor.DATETIME_FORMATS:
            try:
                return datetime.strptime(str(val).strip(), fmt)
            except ValueError:
                continue
        return None

    def load_data(self, pnr_df, target_df, available_df):
        self._load_cancelled_flights(target_df)
        self._load_available_flights(available_df)
        self._load_all_passengers(pnr_df)
        self._build_connection_groups()
        logger.info(
            "Loaded: %d affected, %d non-affected, %d cancelled, "
            "%d available flights",
            len(self.affected_passengers),
            len(self.non_affected_passengers),
            len(self.cancelled_flights),
            len(self.available_flights),
        )

    def _load_cancelled_flights(self, df):
        for _, row in df.iterrows():
            dep_key = str(row["DEP_KEY"]).strip()
            self.cancelled_flights[dep_key] = {
                "dep_key": dep_key,
                "orig_cd": str(row["ORIG_CD"]).strip(),
                "dest_cd": str(row["DEST_CD"]).strip(),
                "flt_num": int(row["FLT_NUM"]),
                "dep_dtml": self._parse_datetime(row.get("DEP_DTML")),
                "arr_dtml": self._parse_datetime(row.get("ARR_DTML")),
            }

    def _load_available_flights(self, df):
        for idx, row in df.iterrows():
            self.available_flights.append(
                Flight(
                    dep_key=str(row["DEP_KEY"]).strip(),
                    dep_dt=str(row.get("DEP_DT", "")),
                    orig_cd=str(row["ORIG_CD"]).strip(),
                    dest_cd=str(row["DEST_CD"]).strip(),
                    flt_num=int(row["FLT_NUM"]),
                    dep_dtml=self._parse_datetime(row.get("DEP_DTML")),
                    arr_dtml=self._parse_datetime(row.get("ARR_DTML")),
                    dep_dtmz=self._parse_datetime(row.get("DEP_DTMZ")),
                    arr_dtmz=self._parse_datetime(row.get("ARR_DTMZ")),
                    c_cap_cnt=int(row.get("C_CAP_CNT", 0)),
                    c_aul_cnt=int(row.get("C_AUL_CNT", 0)),
                    c_pax_cnt=int(row.get("C_PAX_CNT", 0)),
                    c_avail_cnt=int(row.get("C_AVAIL_CNT", 0)),
                    y_cap_cnt=int(row.get("Y_CAP_CNT", 0)),
                    y_aul_cnt=int(row.get("Y_AUL_CNT", 0)),
                    y_pax_cnt=int(row.get("Y_PAX_CNT", 0)),
                    y_avail_cnt=int(row.get("Y_AVAIL_CNT", 0)),
                    original_index=idx,
                )
            )

    def _load_all_passengers(self, df):
        cancelled_keys = set(self.cancelled_flights.keys())
        for idx, row in df.iterrows():
            dep_key = str(row["DEP_KEY"]).strip()
            is_affected = dep_key in cancelled_keys
            pax = Passenger(
                recloc=str(row["RECLOC"]).strip(),
                cabin_cd=str(row.get("CABIN_CD", "Y")).strip().upper(),
                cos_cd=int(row.get("COS_CD", 0)),
                orig_cd=str(row["ORIG_CD"]).strip(),
                dest_cd=str(row["DEST_CD"]).strip(),
                dep_key=dep_key,
                dep_dtml=self._parse_datetime(row.get("DEP_DTML")),
                arr_dtml=self._parse_datetime(row.get("ARR_DTML")),
                dep_dtmz=self._parse_datetime(row.get("DEP_DTMZ")),
                arr_dtmz=self._parse_datetime(row.get("ARR_DTMZ")),
                od_broken_ind=int(row.get("OD_BROKEN_IND", 1)),
                pax_cnt=int(row.get("PAX_CNT", 1)),
                cvm=float(row.get("CVM", 0.0)),
                conn_time_mins=float(row.get("CONN_TIME_MINS", 0)),
                oper_od_orig_cd=str(row.get("OPER_OD_ORIG_CD", "")).strip(),
                oper_od_dest_cd=str(row.get("OPER_OD_DEST_CD", "")).strip(),
                flt_num=int(row.get("FLT_NUM", 0)),
                is_affected=is_affected,
                original_index=idx,
            )
            self.all_passengers.append(pax)
            if is_affected:
                self.affected_passengers.append(pax)
            else:
                self.non_affected_passengers.append(pax)

    def _build_connection_groups(self):
        recloc_groups: Dict[str, List[Passenger]] = {}
        for pax in self.all_passengers:
            recloc_groups.setdefault(pax.recloc, []).append(pax)
        gid = 0
        for recloc, pax_list in recloc_groups.items():
            # Group ALL multi-segment PNRs, regardless of od_broken_ind.
            if len(pax_list) > 1:
                g = f"CONN_{gid}"
                for p in pax_list:
                    p.connection_group_id = g
                self.connection_groups[g] = pax_list
                gid += 1
