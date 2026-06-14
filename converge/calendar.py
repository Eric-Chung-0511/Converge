"""
Working-day to calendar-date conversion utilities.

The simulation engine operates entirely in working-day units relative to a
project start day (day 0). This module provides an *optional* presentation
layer that converts working days to calendar dates. It is intentionally
simple — correctness of the Monte Carlo math never depends on calendar logic.

All published-benchmark validation (Hulett 1996) is checked in
working-day / probability terms only, never against region-specific calendar
dates which would introduce untestable assumptions.

Calendar convention
-------------------
An activity with ES=s and EF=s+D occupies the continuous interval [s, s+D]
in working-day space.  Mapped to calendar dates:

    First day worked  = add_working_days(project_start, s)
                       (ES is the offset to the start day itself)
    Last day worked   = add_working_days(project_start, s + D - 1)
                      = add_working_days(project_start, ef - 1)

Example: project_start = Mon 2026-06-01, A duration = 2 days.
    ES(A) = 0, EF(A) = 2.
    First day = add_working_days(2026-06-01, 0) = 2026-06-01 (Mon)
    Last day  = add_working_days(2026-06-01, 1) = 2026-06-02 (Tue)  ← ef - 1 = 1

B has FS+0 from A, duration = 3 days.
    ES(B) = EF(A) = 2, EF(B) = 5.
    First day = add_working_days(2026-06-01, 2) = 2026-06-03 (Wed)
    Last day  = add_working_days(2026-06-01, 4) = 2026-06-05 (Fri)  ← ef - 1 = 4

Use es_to_date() / ef_to_finish_date() for activity display to get this right.
Use working_day_to_date() for general percentile values (rounds up conservatively).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional


def add_working_days(
    start: date,
    working_days: int,
    holidays: Optional[list[date]] = None,
) -> date:
    """
    Advance *start* by *working_days* working days, skipping weekends and
    any supplied holidays.

    Parameters
    ----------
    start        : The reference calendar date (day 0 in the simulation).
    working_days : Number of working days to advance (non-negative integer).
    holidays     : Optional list of calendar dates to treat as non-working.
                   Typically public holidays; must not include weekends.

    Returns
    -------
    Calendar date corresponding to the working-day offset.
    """
    if working_days < 0:
        raise ValueError("working_days must be non-negative.")
    holiday_set: set[date] = set(holidays) if holidays else set()
    current = start
    remaining = int(working_days)
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in holiday_set:
            remaining -= 1
    return current


def working_days_between(
    start: date,
    end: date,
    holidays: Optional[list[date]] = None,
) -> int:
    """
    Count the number of working days between *start* (inclusive) and *end*
    (exclusive), skipping weekends and supplied holidays.
    """
    if end < start:
        raise ValueError("end must be >= start.")
    holiday_set: set[date] = set(holidays) if holidays else set()
    count = 0
    current = start
    while current < end:
        if current.weekday() < 5 and current not in holiday_set:
            count += 1
        current += timedelta(days=1)
    return count


def working_day_to_date(
    project_start: date,
    working_day: float,
    holidays: Optional[list[date]] = None,
) -> date:
    """
    Convert a general simulation working-day value (e.g. P50 completion) to
    a calendar date.

    Fractional values are rounded up (conservative). This is intended for
    percentile outputs, not for computing the last day an activity is worked.
    For that use ef_to_finish_date().
    """
    return add_working_days(project_start, math.ceil(working_day), holidays)


def es_to_date(
    project_start: date,
    es: float,
    holidays: Optional[list[date]] = None,
) -> date:
    """
    Convert an Early Start working-day offset to the calendar date on which
    the activity begins.

    ES=0  → project_start itself (first working day of the project).
    ES=2  → add_working_days(project_start, 2)
    """
    return add_working_days(project_start, int(es), holidays)


def ef_to_finish_date(
    project_start: date,
    ef: float,
    holidays: Optional[list[date]] = None,
) -> date:
    """
    Convert an Early Finish working-day offset to the *last calendar date on
    which the activity is worked* (inclusive).

    Convention: EF = ES + D, and the activity occupies days [ES, EF).
    The last day worked is therefore add_working_days(project_start, ef - 1).

    Example: project_start=Mon 2026-06-01, ef=5 (3-day activity starting Wed).
        Last day = add_working_days(2026-06-01, 4) = 2026-06-05 (Fri).
    """
    return add_working_days(project_start, max(0, math.ceil(ef) - 1), holidays)
