"""Domain data structures for passengers and flights."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Passenger:
    recloc: str
    cabin_cd: str
    cos_cd: int
    orig_cd: str
    dest_cd: str
    dep_key: str
    dep_dtml: Optional[datetime]
    arr_dtml: Optional[datetime]
    dep_dtmz: Optional[datetime]
    arr_dtmz: Optional[datetime]
    od_broken_ind: int
    pax_cnt: int
    cvm: float
    conn_time_mins: float
    oper_od_orig_cd: str
    oper_od_dest_cd: str
    flt_num: int
    is_affected: bool = True
    original_index: int = -1
    connection_group_id: Optional[str] = None
    has_direct_option: bool = False  # Set during preprocessing

    @property
    def route_key(self) -> str:
        return f"{self.orig_cd}-{self.dest_cd}"

    @property
    def original_duration_mins(self) -> Optional[float]:
        if self.dep_dtml and self.arr_dtml:
            return (self.arr_dtml - self.dep_dtml).total_seconds() / 60.0
        return None


@dataclass
class Flight:
    dep_key: str
    dep_dt: str
    orig_cd: str
    dest_cd: str
    flt_num: int
    dep_dtml: Optional[datetime]
    arr_dtml: Optional[datetime]
    dep_dtmz: Optional[datetime]
    arr_dtmz: Optional[datetime]
    c_cap_cnt: int
    c_aul_cnt: int
    c_pax_cnt: int
    c_avail_cnt: int
    y_cap_cnt: int
    y_aul_cnt: int
    y_pax_cnt: int
    y_avail_cnt: int
    original_index: int = -1

    @property
    def route_key(self) -> str:
        return f"{self.orig_cd}-{self.dest_cd}"

    def available_seats(self, cabin: str) -> int:
        return self.c_avail_cnt if cabin == "C" else self.y_avail_cnt

    def max_capacity(self, cabin: str, overbooking_frac: float = 0.0) -> int:
        if cabin == "C":
            return int(self.c_aul_cnt * (1 + overbooking_frac))
        return int(self.y_aul_cnt * (1 + overbooking_frac))

    @property
    def duration_mins(self) -> Optional[float]:
        if self.dep_dtml and self.arr_dtml:
            return (self.arr_dtml - self.dep_dtml).total_seconds() / 60.0
        return None
