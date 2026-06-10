import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from betmobile_settings import DB_CONFIG, DB_DSN

# ==========================================
# BASELINE MODEL SETTINGS
# ==========================================

ACTIVE_EXPERIMENTAL_FEATURES = {
    "form": False,
    "standings": False,
}

# -----------------------------------------------------
# OUTPUT ROOT
# -----------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------
# ECI RULE PARAMETERS (OVERSCHREVEN DOOR AUTOTUNER)
# -----------------------------------------------------
ECI_RULE_PARAMS = {
    "max_odds": 4.0,
    "min_drift_abs": 0,
    "min_odds": 1.4,
    "min_prob": 0.52,
    "min_rating_gap": 0,
    "min_snapshots": 7,
    "min_value": 1.04
}

RULE_MIN_PROB       = ECI_RULE_PARAMS["min_prob"]
RULE_MIN_VALUE      = ECI_RULE_PARAMS["min_value"]
RULE_MIN_RATING_GAP = ECI_RULE_PARAMS["min_rating_gap"]
RULE_MIN_ODDS       = ECI_RULE_PARAMS["min_odds"]
RULE_MAX_ODDS       = ECI_RULE_PARAMS["max_odds"]
RULE_MIN_SNAPSHOTS  = ECI_RULE_PARAMS["min_snapshots"]
RULE_MIN_DRIFT_ABS  = ECI_RULE_PARAMS["min_drift_abs"]

# Gebruik current drift table (NOW)
USE_CUTOFF_FEATURES = False

# -----------------------------------------------------
# ECI v4.1 — ADDITIONELE ENGINE SETTINGS
# -----------------------------------------------------

# 1) Drift / market support settings
# Negatieve drift = odds dalen = markt beweegt richting die kant
DRIFT_SUPPORT_THRESHOLD = -0.03   # gunstige drift
DRIFT_OPPOSE_THRESHOLD  =  0.03   # ongunstige drift

DRIFT_SUPPORT_BONUS     = 0.10
DRIFT_OPPOSE_PENALTY    = 0.10

SNAP_BONUS_THRESHOLD    = 15
SNAP_BONUS              = 0.05

RANGE_PENALTY_THRESHOLD = 0.50
RANGE_PENALTY           = 0.05

STALE_PENALTY_THRESHOLD = 8.0
STALE_PENALTY           = 0.05

KICKOFF_SOON_HOURS      = 12.0
KICKOFF_SOON_BONUS      = 0.05

KICKOFF_VERY_SOON_HOURS = 4.0
KICKOFF_VERY_SOON_BONUS = 0.05

STALE_NEAR_KICKOFF_HOURS = 2.0
STALE_DAY_KICKOFF_HOURS  = 6.0

# 1b) Extra drift quality settings
DRIFT_STRONG_THRESHOLD      = 0.08
DRIFT_STRONG_BONUS          = 0.05
DRIFT_STRONG_PENALTY        = 0.05

DRIFT_CONSISTENCY_THRESHOLD = 0.20
DRIFT_CONSISTENCY_BONUS     = 0.05
DRIFT_CONSISTENCY_PENALTY   = 0.05

DRIFT_NOISE_MULTIPLIER      = 3.0
DRIFT_NOISE_PENALTY         = 0.05

LAST_MOVE_SUPPORT_THRESHOLD = -0.02
LAST_MOVE_OPPOSE_THRESHOLD  =  0.02
LAST_MOVE_BONUS             = 0.05
LAST_MOVE_PENALTY           = 0.05

RECENT24_SUPPORT_THRESHOLD  = -0.04
RECENT24_OPPOSE_THRESHOLD   =  0.04
RECENT24_BONUS              = 0.05
RECENT24_PENALTY            = 0.05

# 2) Calibration settings
CALIB_MIN_PICKS = 200
CALIB_N_BANDS = 5
CALIB_MIN_BETS_PER_BAND = 30
CALIB_ROI_CLIP = 0.20

# 3) **NIEUW** — Hard pick-filter
MIN_STRENGTH = 1.5

# =========================
# SECONDARY (SINGLE_FAIL) SETTINGS
# =========================

ENABLE_SECONDARY_PICKS = True

SECONDARY_ALLOWED_FAIL = "value"

# hoe ver onder min_value mag hij nog zitten
SECONDARY_VALUE_TOLERANCE = 0.04  
# voorbeeld: min_value=1.04 → dan tot 0.98 toegestaan

# extra veiligheid
SECONDARY_MIN_STRENGTH = 1.8

# optioneel: lagere confidence
SECONDARY_MIN_PROB = 0.50
