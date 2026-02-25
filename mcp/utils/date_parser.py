"""
Date Parser Utilities for MCP Tools

Parses natural language date expressions into datetime objects.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple


def parse_relative_date(date_str: str) -> Optional[datetime]:
    """
    Parse a relative date string into a datetime.

    Supports formats like:
    - ISO format: "2024-01-15", "2024-01-15T10:30:00"
    - Relative: "today", "yesterday", "last week", "30 days ago"
    - Natural: "last 7 days", "past month", "this week"

    Args:
        date_str: Date string to parse

    Returns:
        Parsed datetime or None if parsing fails
    """
    if not date_str:
        return None

    date_str = date_str.strip().lower()
    now = datetime.utcnow()

    # Try ISO format first
    iso_result = _try_parse_iso(date_str)
    if iso_result:
        return iso_result

    # Handle relative expressions
    if date_str == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    if date_str == "yesterday":
        return (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    if date_str == "tomorrow":
        return (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    if date_str in ("now", "current"):
        return now

    # "N days ago", "N weeks ago", etc.
    ago_match = re.match(
        r"(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*ago",
        date_str,
    )
    if ago_match:
        amount = int(ago_match.group(1))
        unit = ago_match.group(2).rstrip("s")  # Normalize to singular
        return _subtract_time(now, amount, unit)

    # "last N days", "past N weeks", etc.
    last_match = re.match(
        r"(?:last|past)\s*(\d+)\s*(day|days|week|weeks|month|months)",
        date_str,
    )
    if last_match:
        amount = int(last_match.group(1))
        unit = last_match.group(2).rstrip("s")
        return _subtract_time(now, amount, unit)

    # "last week", "last month", "last year"
    if date_str == "last week":
        return now - timedelta(weeks=1)
    if date_str == "last month":
        return _subtract_months(now, 1)
    if date_str == "last year":
        return _subtract_months(now, 12)

    # "this week", "this month"
    if date_str == "this week":
        return now - timedelta(days=now.weekday())
    if date_str == "this month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if date_str == "this year":
        return now.replace(
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )

    # "start of month", "end of month"
    if date_str in ("start of month", "beginning of month"):
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if date_str == "end of month":
        next_month = now.replace(day=28) + timedelta(days=4)
        return next_month.replace(day=1) - timedelta(days=1)

    return None


def _try_parse_iso(date_str: str) -> Optional[datetime]:
    """Try to parse as ISO format date"""
    # Common ISO formats to try
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]

    # Handle uppercase for ISO format
    date_str_upper = date_str.upper()

    for fmt in formats:
        try:
            return datetime.strptime(date_str_upper, fmt)
        except ValueError:
            continue

    return None


def _subtract_time(dt: datetime, amount: int, unit: str) -> datetime:
    """Subtract time from a datetime"""
    if unit == "day":
        return dt - timedelta(days=amount)
    elif unit == "week":
        return dt - timedelta(weeks=amount)
    elif unit == "month":
        return _subtract_months(dt, amount)
    elif unit == "year":
        return _subtract_months(dt, amount * 12)
    return dt


def _subtract_months(dt: datetime, months: int) -> datetime:
    """Subtract months from a datetime"""
    year = dt.year
    month = dt.month - months

    while month <= 0:
        month += 12
        year -= 1

    # Handle day overflow (e.g., Jan 31 - 1 month = Dec 31, not Dec 28)
    day = min(dt.day, _days_in_month(year, month))

    return dt.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    """Get number of days in a month"""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return 29
        return 28
    return 30


def parse_date_range(
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    days: Optional[int] = None,
) -> Tuple[datetime, datetime]:
    """
    Parse a date range from strings or default to last N days.

    Args:
        start_str: Start date string (optional)
        end_str: End date string (optional)
        days: Default number of days if no dates provided

    Returns:
        Tuple of (start_date, end_date)
    """
    now = datetime.utcnow()

    # Parse end date first (defaults to now)
    if end_str:
        end_date = parse_relative_date(end_str)
        if not end_date:
            end_date = now
    else:
        end_date = now

    # Parse start date
    if start_str:
        start_date = parse_relative_date(start_str)
        if not start_date:
            # Default to 30 days before end
            start_date = end_date - timedelta(days=days or 30)
    elif days:
        start_date = end_date - timedelta(days=days)
    else:
        # Default to 30 days
        start_date = end_date - timedelta(days=30)

    # Ensure start is before end
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    return start_date, end_date


def format_date_for_display(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime for display in responses"""
    if not dt:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_date_range_description(start: datetime, end: datetime) -> str:
    """Get a human-readable description of a date range"""
    days = (end - start).days

    if days == 0:
        return "today"
    elif days == 1:
        return "last 1 day"
    elif days == 7:
        return "last week"
    elif days == 30:
        return "last 30 days"
    elif days == 90:
        return "last 3 months"
    else:
        return f"from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

