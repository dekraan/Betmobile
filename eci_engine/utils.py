import pandas as pd
import numpy as np

def _json_default(o):
    import numpy as np
    import pandas as pd
    from datetime import datetime, date
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (datetime, date, pd.Timestamp)):
        return o.isoformat()
    # Fallback voor eventuele resterende niet-standaard types
    if hasattr(o, "left") and hasattr(o, "right"):  # pandas.Interval
        return {"left": float(o.left), "right": float(o.right), "closed": str(o.closed)}
    return str(o)

def safe_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def make_strength_bucket(x):
    if x is None or pd.isna(x):
        return "UNKNOWN"
    if x < 1:
        return "<1"
    if x < 1.5:
        return "1-1.5"
    if x < 2:
        return "1.5-2"
    if x < 3:
        return "2-3"
    return "3+"

def calc_drift_consistency(drift_pct, drift_range):
    if pd.isna(drift_pct) or pd.isna(drift_range):
        return None
    base = max(float(drift_range), 0.01)
    return abs(float(drift_pct)) / base

def get_raw_strength(row, side: str) -> float:
    if side == "HOME":
        return safe_float(row.get("RawStrength_Home_All"), 0) or 0
    if side == "AWAY":
        return safe_float(row.get("RawStrength_Away_All"), 0) or 0
    return 0.0


def choose_relevant_side(
    row,
    home_condition: bool,
    away_condition: bool,
    tie_break_on_strength: bool = True,
):
    """
    Kies de relevante kant:
    - alleen HOME als alleen home_condition waar is
    - alleen AWAY als alleen away_condition waar is
    - als beide waar zijn: kies hoogste RawStrength_*_All
    """
    if home_condition and not away_condition:
        return "HOME"
    if away_condition and not home_condition:
        return "AWAY"
    if home_condition and away_condition:
        if not tie_break_on_strength:
            return None
        home_strength = get_raw_strength(row, "HOME")
        away_strength = get_raw_strength(row, "AWAY")
        return "HOME" if home_strength >= away_strength else "AWAY"
    return None