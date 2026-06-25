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
# ECI RULE PARAMETERS
# -----------------------------------------------------
# Gewijzigd op basis van:
#   - Unibet/Oddspedia benchmark (volledig seizoen 2025/2026)
#   - Bet365 picks_evaluated (april-juni 2026)
#
# Belangrijkste wijzigingen t.o.v. oude config:
#   min_rating_gap: 0 → 500   (grootste verbetering)
#   max_odds:       4.0 → 2.50 (odds boven 2.50 structureel verlieslatend)
#   min_odds:       1.40 → 1.50 (onder 1.50 te weinig value)
#   min_prob:       0.52 → 0.58 (iets strenger voor stabiliteit)
#   min_value:      1.04 → 1.02 (iets soepeler om volume te houden)
# -----------------------------------------------------
ECI_RULE_PARAMS = {
    "min_prob":        0.58,
    "min_value":       1.02,
    "min_rating_gap":  500,
    "min_odds":        1.50,
    "max_odds":        2.50,
    "min_snapshots":   7,
    "min_drift_abs":   0,
}

RULE_MIN_PROB       = ECI_RULE_PARAMS["min_prob"]
RULE_MIN_VALUE      = ECI_RULE_PARAMS["min_value"]
RULE_MIN_RATING_GAP = ECI_RULE_PARAMS["min_rating_gap"]
RULE_MIN_ODDS       = ECI_RULE_PARAMS["min_odds"]
RULE_MAX_ODDS       = ECI_RULE_PARAMS["max_odds"]
RULE_MIN_SNAPSHOTS  = ECI_RULE_PARAMS["min_snapshots"]
RULE_MIN_DRIFT_ABS  = ECI_RULE_PARAMS["min_drift_abs"]

USE_CUTOFF_FEATURES = False

# -----------------------------------------------------
# ECI v4.1 — DRIFT & MARKET SETTINGS (ongewijzigd)
# -----------------------------------------------------

DRIFT_SUPPORT_THRESHOLD = -0.03
DRIFT_OPPOSE_THRESHOLD  =  0.03

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

# -----------------------------------------------------
# CALIBRATION
# -----------------------------------------------------
CALIB_MIN_PICKS      = 200
CALIB_N_BANDS        = 5
CALIB_MIN_BETS_PER_BAND = 30
CALIB_ROI_CLIP       = 0.20

# -----------------------------------------------------
# PICK FILTERS
# -----------------------------------------------------
# MIN_STRENGTH verhoogd: met strakkere regels zijn
# zwakke picks minder interessant
MIN_STRENGTH = 1.8

# -----------------------------------------------------
# SECONDARY PICKS
# Iets soepeler: secondary picks presteren goed
# (77-78% hitrate in Bet365 data)
# -----------------------------------------------------
ENABLE_SECONDARY_PICKS = True

SECONDARY_ALLOWED_FAIL    = "value"
SECONDARY_VALUE_TOLERANCE = 0.05   # was 0.04 — iets ruimer
SECONDARY_MIN_STRENGTH    = 1.6    # was 1.8 — iets soepeler
SECONDARY_MIN_PROB        = 0.52   # ongewijzigd