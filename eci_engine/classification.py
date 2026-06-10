import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from utils import safe_float


# =====================================================================
# PICK CLASSIFICATION / TIERING
# =====================================================================
# Doel:
# - GEEN picklogica veranderen.
# - Alleen extra labels geven aan picks die al door build_picks() zijn gemaakt.
# - Later dynamisch te voeden met strong_segments.json / danger_segments.json.
#
# Outputkolommen:
# - pick_tier: A+, A, B, C, X
# - pick_stars: 5, 4, 3, 2, 1
# - sector_tags: positieve/verklarende tags
# - danger_tags: negatieve/waarschuwings-tags
# - classification_reason: korte uitleg voor Excel/Discord/research
# - passes_danger_combo_v2
# - passes_danger_combo_v2_no_longshots


SEGMENT_DIR = OUTPUT_DIR / "research"
STRONG_SEGMENTS_PATH = SEGMENT_DIR / "strong_segments.json"
DANGER_SEGMENTS_PATH = SEGMENT_DIR / "danger_segments.json"


def _as_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _selected_value(row: pd.Series, home_col: str, away_col: str, draw_col: str | None = None):
    sel = _as_str(row.get("Selection")).upper()
    if sel == "HOME":
        return row.get(home_col)
    if sel == "AWAY":
        return row.get(away_col)
    if sel == "DRAW" and draw_col:
        return row.get(draw_col)
    return np.nan


def get_selected_odds(row: pd.Series) -> float | None:
    return safe_float(_selected_value(row, "odds_home", "odds_away", "odds_draw"))


def get_selected_prob(row: pd.Series) -> float | None:
    return safe_float(_selected_value(row, "Home Prob", "Away Prob", "Draw Prob"))


def get_selected_value_score(row: pd.Series) -> float | None:
    return safe_float(_selected_value(row, "bet_home", "bet_away", "bet_draw"))


def get_selected_drift_pct(row: pd.Series) -> float | None:
    return safe_float(_selected_value(row, "home_drift_pct", "away_drift_pct"))


def get_selected_strength(row: pd.Series) -> float | None:
    pick_type = _as_str(row.get("PickType")).upper()

    if pick_type == "SECONDARY":
        return safe_float(row.get("SecondaryStrength"))

    # MAIN: pak bij voorkeur de geselecteerde/calibrated strength.
    for col in ["SelectedStrength", "SelectedAdjStrength"]:
        val = safe_float(row.get(col))
        if val is not None:
            return val

    sel = _as_str(row.get("Selection")).upper()
    if sel == "HOME":
        for col in ["RuleStrengthCalibrated_Home", "RuleStrengthAdj_Home", "RawStrength_Home"]:
            val = safe_float(row.get(col))
            if val is not None:
                return val
    if sel == "AWAY":
        for col in ["RuleStrengthCalibrated_Away", "RuleStrengthAdj_Away", "RawStrength_Away"]:
            val = safe_float(row.get(col))
            if val is not None:
                return val

    return safe_float(row.get("RuleStrengthCalibrated")) or safe_float(row.get("RuleStrengthAdj"))


def bucket_odds(odds: float | None) -> str:
    if odds is None or pd.isna(odds):
        return "UNKNOWN"
    if odds < 1.4:
        return "1.0-1.4"
    if odds < 1.6:
        return "1.4-1.6"
    if odds < 1.8:
        return "1.6-1.8"
    if odds < 2.2:
        return "1.8-2.2"
    if odds < 3.0:
        return "2.2-3.0"
    return "3.0+"


def bucket_prob(prob: float | None) -> str:
    if prob is None or pd.isna(prob):
        return "UNKNOWN"
    if prob < 0.52:
        return "<52%"
    if prob < 0.55:
        return "52-55%"
    if prob < 0.60:
        return "55-60%"
    if prob < 0.65:
        return "60-65%"
    if prob < 0.70:
        return "65-70%"
    return "70%+"


def bucket_rating_gap(rating_gap: float | None) -> str:
    if rating_gap is None or pd.isna(rating_gap):
        return "UNKNOWN"
    if rating_gap < 100:
        return "0-100"
    if rating_gap < 250:
        return "100-250"
    if rating_gap < 500:
        return "250-500"
    if rating_gap < 1000:
        return "500-1000"
    if rating_gap < 5000:
        return "1000-5000"
    return "5000+"


def bucket_value(value_score: float | None) -> str:
    if value_score is None or pd.isna(value_score):
        return "UNKNOWN"
    if value_score < 0.95:
        return "<0.95"
    if value_score < 1.00:
        return "0.95-1.00"
    if value_score < 1.04:
        return "1.00-1.04"
    if value_score < 1.08:
        return "1.04-1.08"
    if value_score < 1.15:
        return "1.08-1.15"
    if value_score < 1.30:
        return "1.15-1.30"
    return "1.30+"

def bucket_snapshots(snapshots: float | None) -> str:
    if snapshots is None or pd.isna(snapshots):
        return "UNKNOWN"
    if snapshots <= 3:
        return "0-3"
    if snapshots <= 6:
        return "4-6"
    if snapshots <= 10:
        return "7-10"
    if snapshots <= 15:
        return "11-15"
    if snapshots <= 25:
        return "16-25"
    if snapshots <= 50:
        return "26-50"
    return "50+"


def bucket_strength(strength: float | None) -> str:
    if strength is None or pd.isna(strength):
        return "UNKNOWN"
    if strength < 1:
        return "<1"
    if strength < 1.5:
        return "1-1.5"
    if strength < 2:
        return "1.5-2"
    if strength < 3:
        return "2-3"
    return "3+"


def bucket_drift(drift_pct: float | None) -> str:
    if drift_pct is None or pd.isna(drift_pct):
        return "UNKNOWN"
    if drift_pct <= -0.10:
        return "<=-10%"
    if drift_pct <= -0.05:
        return "-10/-5%"
    if drift_pct <= -0.03:
        return "-5/-3%"
    if drift_pct <= 0.00:
        return "-3/0%"
    if drift_pct <= 0.03:
        return "0/3%"
    if drift_pct <= 0.05:
        return "3/5%"
    if drift_pct <= 0.10:
        return "5/10%"
    return ">10%"


def get_date_features(row: pd.Series) -> tuple[str, str, str]:
    dt = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.isna(dt):
        return "UNKNOWN", "UNKNOWN", "unknown"

    month = dt.to_period("M").strftime("%Y-%m")
    weekday = dt.day_name()
    month_num = dt.month

    if month_num in [8, 9]:
        season_phase = "early"
    elif month_num in [10, 11, 12, 1, 2]:
        season_phase = "mid"
    elif month_num in [3, 4, 5]:
        season_phase = "late"
    elif month_num in [6, 7]:
        season_phase = "summer"
    else:
        season_phase = "unknown"

    return month, weekday, season_phase

def market_support_label(drift_pct: float | None) -> str:
    if drift_pct is None or pd.isna(drift_pct):
        return "NEUTRAL"
    if drift_pct <= -0.03:
        return "SUPPORT"
    if drift_pct >= 0.03:
        return "AGAINST"
    return "NEUTRAL"


def load_segment_config(path: Path) -> list[dict[str, Any]]:
    """
    Segmentconfig is optioneel.
    Voorbeeld:
    [
      {
        "name": "prob65_70_odds16_18_value115_130",
        "conditions": {
          "prob_bucket": "65-70%",
          "odds_bucket": "1.6-1.8",
          "value_bucket": "1.15-1.30"
        }
      }
    ]
    """
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[classification][WARN] Kon {path.name} niet lezen: {e}")
        return []

    if isinstance(data, dict):
        data = data.get("segments", [])

    if not isinstance(data, list):
        print(f"[classification][WARN] {path.name} bevat geen lijst met segmenten.")
        return []

    return [x for x in data if isinstance(x, dict)]


def _matches_conditions(features: dict[str, Any], conditions: dict[str, Any]) -> bool:
    for key, expected in conditions.items():
        actual = features.get(key)

        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False

    return True


def match_segment_config(features: dict[str, Any], segments: list[dict[str, Any]]) -> list[str]:
    matches = []

    for seg in segments:
        name = seg.get("name") or seg.get("segment") or "unnamed_segment"
        conditions = seg.get("conditions", {})
        if not isinstance(conditions, dict):
            continue

        if _matches_conditions(features, conditions):
            matches.append(str(name))

    return matches


def calc_research_experiment_flags(
    pick_type: str,
    selected_odds: float | None,
    selected_prob: float | None,
    selected_strength: float | None,
    rating_gap: float | None,
) -> tuple[bool, bool]:
    """
    Zelfde idee als research_backtest_v6 danger_combo_v2.
    Dit verandert GEEN pickselectie; het labelt alleen.
    """
    selected_odds = safe_float(selected_odds)
    selected_prob = safe_float(selected_prob)
    selected_strength = safe_float(selected_strength)
    rating_gap = safe_float(rating_gap)

    if selected_odds is None or selected_prob is None or rating_gap is None:
        return False, False

    is_main = _as_str(pick_type).upper() == "MAIN"

    main_strength_bad = (
        is_main
        and selected_strength is not None
        and selected_strength < 2.0
    )

    odds_22_30_bad = 2.2 <= selected_odds < 3.0
    prob_55_60_bad = 0.55 <= selected_prob < 0.60
    rating_gap_bad = rating_gap < 250
    low_odds_high_prob_bad = 1.4 <= selected_odds < 1.6 and selected_prob >= 0.70

    passes_v2 = not (
        main_strength_bad
        or odds_22_30_bad
        or prob_55_60_bad
        or rating_gap_bad
        or low_odds_high_prob_bad
    )

    passes_v2_no_longshots = passes_v2 and selected_odds < 3.0

    return bool(passes_v2), bool(passes_v2_no_longshots)


def classify_pick(
    row: pd.Series,
    strong_segments: list[dict[str, Any]] | None = None,
    danger_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    strong_segments = strong_segments or []
    danger_segments = danger_segments or []

    pick_type = _as_str(row.get("PickType")).upper()
    selection = _as_str(row.get("Selection")).upper()

    odds = get_selected_odds(row)
    prob = get_selected_prob(row)
    value_score = get_selected_value_score(row)
    drift_pct = get_selected_drift_pct(row)
    strength = get_selected_strength(row)
    rating_gap = safe_float(row.get("rating_gap"))
    snapshots = safe_float(row.get("n_snapshots")) or safe_float(row.get("snapshot_count"))

    odds_bucket = bucket_odds(odds)
    prob_bucket = bucket_prob(prob)
    rating_gap_bucket = bucket_rating_gap(rating_gap)
    value_bucket = bucket_value(value_score)
    market_support = market_support_label(drift_pct)

    strength_bucket = bucket_strength(strength)
    snapshot_bucket = bucket_snapshots(snapshots)
    drift_bucket = bucket_drift(drift_pct)
    month, weekday, season_phase = get_date_features(row)
    competition = _as_str(row.get("competition")) or "UNKNOWN"

    features = {
        "pick_type": pick_type,
        "selection": selection,
        "competition": competition,
        "strength_bucket": strength_bucket,
        "odds_bucket": odds_bucket,
        "prob_bucket": prob_bucket,
        "rating_gap_bucket": rating_gap_bucket,
        "snapshot_bucket": snapshot_bucket,
        "market_support": market_support,
        "value_bucket": value_bucket,
        "drift_bucket": drift_bucket,
        "season_phase": season_phase,
        "weekday": weekday,
        "month": month,
    }

    sector_tags: list[str] = []
    danger_tags: list[str] = []

    if pick_type:
        sector_tags.append(f"type:{pick_type}")
    if odds_bucket != "UNKNOWN":
        sector_tags.append(f"odds:{odds_bucket}")
    if prob_bucket != "UNKNOWN":
        sector_tags.append(f"prob:{prob_bucket}")
    if rating_gap_bucket != "UNKNOWN":
        sector_tags.append(f"gap:{rating_gap_bucket}")
    if value_bucket != "UNKNOWN":
        sector_tags.append(f"value:{value_bucket}")
    if market_support != "NEUTRAL":
        sector_tags.append(f"market:{market_support}")
    if competition != "UNKNOWN":
        sector_tags.append(f"competition:{competition}")
    if strength_bucket != "UNKNOWN":
        sector_tags.append(f"strength:{strength_bucket}")
    if snapshot_bucket != "UNKNOWN":
        sector_tags.append(f"snapshots:{snapshot_bucket}")
    if drift_bucket != "UNKNOWN":
        sector_tags.append(f"drift:{drift_bucket}")
    if season_phase != "unknown":
        sector_tags.append(f"season:{season_phase}")
    if weekday != "UNKNOWN":
        sector_tags.append(f"weekday:{weekday}")
    if month != "UNKNOWN":
        sector_tags.append(f"month:{month}")

    # -------------------------------------------------
    # Vaste danger-regels uit huidige research
    # -------------------------------------------------
    if pick_type == "MAIN" and strength is not None and strength < 2.0:
        danger_tags.append("MAIN_strength_below_2")

    if odds is not None and 2.2 <= odds < 3.0:
        danger_tags.append("odds_2.2_3.0")

    if prob is not None and 0.55 <= prob < 0.60:
        danger_tags.append("prob_55_60")

    if rating_gap is not None and rating_gap < 250:
        danger_tags.append("rating_gap_below_250")

    if odds is not None and prob is not None and 1.4 <= odds < 1.6 and prob >= 0.70:
        danger_tags.append("low_odds_high_prob_70plus")

    if market_support == "SUPPORT" and odds is not None and 2.2 <= odds < 3.0:
        danger_tags.append("support_odds_2.2_3.0")

    # Config-gestuurde danger/strong segmenten
    matched_danger = match_segment_config(features, danger_segments)
    matched_strong = match_segment_config(features, strong_segments)

    for name in matched_danger:
        danger_tags.append(f"dynamic_danger:{name}")

    for name in matched_strong:
        sector_tags.append(f"dynamic_strong:{name}")

    passes_v2, passes_v2_nl = calc_research_experiment_flags(
        pick_type=pick_type,
        selected_odds=odds,
        selected_prob=prob,
        selected_strength=strength,
        rating_gap=rating_gap,
    )

    if passes_v2:
        sector_tags.append("passes:danger_combo_v2")
    if passes_v2_nl:
        sector_tags.append("passes:danger_combo_v2_no_longshots")

    # -------------------------------------------------
    # Tiering
    # -------------------------------------------------
    # X = danger / alleen loggen
    # A+ = safe + dynamisch sterk segment
    # A = safe + passes danger_combo_v2_no_longshots
    # B = gewone MAIN
    # C = SECONDARY / research
    fixed_danger_count = sum(
        1
        for tag in danger_tags
        if not tag.startswith("dynamic_danger:")
    )

    dynamic_danger_count = len(matched_danger)

    danger_count = fixed_danger_count
    strong_count = len(matched_strong)

    if danger_count >= 2:
        tier = "X"
        stars = 1
        label = "Danger / only log"

    elif strong_count >= 1 and danger_count == 0 and passes_v2_nl and pick_type == "MAIN":
        tier = "A+"
        stars = 5
        label = "Strong dynamic sector"

    elif strong_count >= 1 and danger_count == 1 and pick_type == "MAIN":
        tier = "A-"
        stars = 4
        label = "Strong but mixed signals"

    elif passes_v2_nl and pick_type == "MAIN":
        tier = "A"
        stars = 4
        label = "Strong baseline sector"

    elif pick_type == "MAIN":
        tier = "B"
        stars = 3
        label = "Regular MAIN"

    elif pick_type == "SECONDARY":
        tier = "C"
        stars = 2
        label = "Secondary / research"

    else:
        tier = "C"
        stars = 2
        label = "Research"

    reason_parts = [
        label,
        f"competition={competition}",
        f"selection={selection}",
        f"odds={odds_bucket}",
        f"prob={prob_bucket}",
        f"gap={rating_gap_bucket}",
        f"value={value_bucket}",
        f"strength={strength_bucket}",
        f"snapshots={snapshot_bucket}",
        f"drift={drift_bucket}",
        f"market={market_support}",
        f"weekday={weekday}",
        f"season={season_phase}",
    ]

    if danger_tags:
        reason_parts.append("danger=" + ",".join(danger_tags))
    if matched_strong:
        reason_parts.append("strong=" + ",".join(matched_strong))

    return {
        "pick_tier": tier,
        "pick_stars": int(stars),
        "sector_tags": ",".join(dict.fromkeys(sector_tags)),
        "danger_tags": ",".join(dict.fromkeys(danger_tags)),
        "classification_reason": " | ".join(reason_parts),
        "selected_odds": odds,
        "selected_prob": prob,
        "selected_value_score": value_score,
        "selected_drift_pct": drift_pct,
        "classification_strength": strength,
        "odds_bucket": odds_bucket,
        "prob_bucket": prob_bucket,
        "rating_gap_bucket": rating_gap_bucket,
        "value_bucket": value_bucket,
        "market_support": market_support,
        "competition_bucket": competition,
        "strength_bucket": strength_bucket,
        "snapshot_bucket": snapshot_bucket,
        "drift_bucket": drift_bucket,
        "season_phase": season_phase,
        "weekday": weekday,
        "month": month,
        "passes_danger_combo_v2": passes_v2,
        "passes_danger_combo_v2_no_longshots": passes_v2_nl,
    }


def classify_picks(picks: pd.DataFrame) -> pd.DataFrame:
    if picks is None or picks.empty:
        return picks

    out = picks.copy()

    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    strong_segments = load_segment_config(STRONG_SEGMENTS_PATH)
    danger_segments = load_segment_config(DANGER_SEGMENTS_PATH)

    rows = [
        classify_pick(row, strong_segments=strong_segments, danger_segments=danger_segments)
        for _, row in out.iterrows()
    ]

    class_df = pd.DataFrame(rows, index=out.index)

    # Niet zomaar bestaande modelkolommen overschrijven, behalve expliciete classificatievelden.
    for col in class_df.columns:
        out[col] = class_df[col]

    # Sorteer voor output: A+ bovenaan, dan A, B, C, X.
    tier_order = {"A+": 0, "A": 1, "A-": 2, "B": 3, "C": 4, "X": 5}
    out["_tier_order"] = out["pick_tier"].map(tier_order).fillna(9)
    sort_strength = pd.to_numeric(out.get("SortStrength"), errors="coerce").fillna(0)
    out["_sort_strength_tmp"] = sort_strength
    out = out.sort_values(["_tier_order", "_sort_strength_tmp"], ascending=[True, False]).drop(
        columns=["_tier_order", "_sort_strength_tmp"]
    )

    return out
