import pandas as pd
import numpy as np

from config import (
    MIN_STRENGTH,
    ENABLE_SECONDARY_PICKS,
    SECONDARY_ALLOWED_FAIL,
    SECONDARY_VALUE_TOLERANCE,
    SECONDARY_MIN_STRENGTH,
    SECONDARY_MIN_PROB,
    RULE_MIN_VALUE,
    RULE_MIN_PROB,
)
from utils import make_strength_bucket


# =====================================================================
# BUILD PICKS (filter op Rule + MIN_STRENGTH + advies HOME/AWAY)
# =====================================================================
def build_picks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    strength_col = "RuleStrengthCalibrated" if "RuleStrengthCalibrated" in df.columns else "RuleStrengthAdj"

    # =========================
    # 1. MAIN picks
    # =========================
    main_picks = df[df["Rule"]].copy()
    if not main_picks.empty:

        def choose_main_selection(r):
            away_ok = bool(r["AwayRule"])
            home_ok = bool(r["HomeRule"])

            away_strength = r.get("RuleStrengthCalibrated_Away", r.get("RuleStrengthAdj_Away", r.get("RawStrength_Away", 0)))
            home_strength = r.get("RuleStrengthCalibrated_Home", r.get("RuleStrengthAdj_Home", r.get("RawStrength_Home", 0)))

            if away_ok and not home_ok:
                return "AWAY"
            if home_ok and not away_ok:
                return "HOME"
            if away_ok and home_ok:
                return "AWAY" if away_strength >= home_strength else "HOME"
            return None

        main_picks["Selection"] = main_picks.apply(choose_main_selection, axis=1)
        main_picks = main_picks[main_picks["Selection"].notna()].copy()

        main_picks["SelectedStrength"] = np.where(
            main_picks["Selection"] == "HOME",
            main_picks.get("RuleStrengthCalibrated_Home", main_picks.get("RuleStrengthAdj_Home", main_picks.get("RawStrength_Home", 0))),
            np.where(
                main_picks["Selection"] == "AWAY",
                main_picks.get("RuleStrengthCalibrated_Away", main_picks.get("RuleStrengthAdj_Away", main_picks.get("RawStrength_Away", 0))),
                np.nan
            )
        )

        main_picks["SelectedRawStrength"] = np.where(
            main_picks["Selection"] == "HOME",
            main_picks.get("RawStrength_Home", 0),
            np.where(
                main_picks["Selection"] == "AWAY",
                main_picks.get("RawStrength_Away", 0),
                np.nan
            )
        )

        main_picks["SelectedAdjStrength"] = np.where(
            main_picks["Selection"] == "HOME",
            main_picks.get("RuleStrengthAdj_Home", main_picks.get("RawStrength_Home", 0)),
            np.where(
                main_picks["Selection"] == "AWAY",
                main_picks.get("RuleStrengthAdj_Away", main_picks.get("RawStrength_Away", 0)),
                np.nan
            )
        )

        # Hard filter pas na side-keuze
        main_picks = main_picks[main_picks["SelectedStrength"] >= MIN_STRENGTH].copy()

        main_picks["Advice"] = main_picks["Selection"]
        main_picks["PickType"] = "MAIN"
        main_picks["rule_passed"] = True
    else:
        main_picks = pd.DataFrame()

    # =========================
    # 2. SECONDARY picks
    # =========================
    secondary_picks = pd.DataFrame()

    if ENABLE_SECONDARY_PICKS:
        sec = df[~df["Rule"]].copy()

        def choose_secondary_selection(r):
            candidates = []

            # HOME secondary
            home_single_value_fail = (
                r["home_fail_count"] == 1 and
                r["home_fail_reasons"] == SECONDARY_ALLOWED_FAIL
            )

            if home_single_value_fail:
                home_value_ok_soft = r["bet_home"] >= (RULE_MIN_VALUE - SECONDARY_VALUE_TOLERANCE)
                home_prob_ok_soft = r["Home Prob"] >= SECONDARY_MIN_PROB
                home_strength_ok = r["RawStrength_Home_All"] >= SECONDARY_MIN_STRENGTH

                if home_value_ok_soft and home_prob_ok_soft and home_strength_ok:
                    candidates.append(("HOME", r["RawStrength_Home_All"]))

            # AWAY secondary
            away_single_value_fail = (
                r["away_fail_count"] == 1 and
                r["away_fail_reasons"] == SECONDARY_ALLOWED_FAIL
            )

            if away_single_value_fail:
                away_value_ok_soft = r["bet_away"] >= (RULE_MIN_VALUE - SECONDARY_VALUE_TOLERANCE)
                away_prob_ok_soft = r["Away Prob"] >= SECONDARY_MIN_PROB
                away_strength_ok = r["RawStrength_Away_All"] >= SECONDARY_MIN_STRENGTH

                if away_value_ok_soft and away_prob_ok_soft and away_strength_ok:
                    candidates.append(("AWAY", r["RawStrength_Away_All"]))

            if not candidates:
                return None

            # kies kandidaat met hoogste ruwe strength
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        if not sec.empty:
            sec["Selection"] = sec.apply(choose_secondary_selection, axis=1)
            sec = sec[sec["Selection"].notna()].copy()

            if not sec.empty:
                sec["Advice"] = sec["Selection"]
                sec["PickType"] = "SECONDARY"
                sec["rule_passed"] = False

                # Geef secondary picks ook een strength-kolom om op te sorteren
                sec["SecondaryStrength"] = np.where(
                    sec["Selection"] == "HOME",
                    sec["RawStrength_Home_All"],
                    sec["RawStrength_Away_All"]
                )

                secondary_picks = sec.copy()

    # =========================
    # 3. Combine
    # =========================
    frames = []

    if not main_picks.empty:
        frames.append(main_picks)

    if not secondary_picks.empty:
        frames.append(secondary_picks)

    if not frames:
        return pd.DataFrame()

    picks = pd.concat(frames, ignore_index=True)

    # Sorteer: MAIN op calibrated strength, SECONDARY op SecondaryStrength
    picks["SortStrength"] = np.where(
        picks["PickType"] == "MAIN",
        picks.get("SelectedStrength", picks[strength_col]),
        picks.get("SecondaryStrength", 0)
    )

    # =========================
    # Extra analysevelden
    # =========================

    # value_edge: altijd t.o.v. MAIN value-grens
    picks["value_edge"] = np.where(
        picks["Selection"] == "HOME",
        picks["bet_home"] - RULE_MIN_VALUE,
        np.where(
            picks["Selection"] == "AWAY",
            picks["bet_away"] - RULE_MIN_VALUE,
            np.nan
        )
    )

    # prob_edge: t.o.v. grens die hoort bij picktype
    prob_floor = np.where(
        picks["PickType"] == "SECONDARY",
        SECONDARY_MIN_PROB,
        RULE_MIN_PROB
    )

    sel_prob = np.where(
        picks["Selection"] == "HOME",
        picks["Home Prob"],
        np.where(
            picks["Selection"] == "AWAY",
            picks["Away Prob"],
            np.nan
        )
    )

    picks["prob_edge"] = sel_prob - prob_floor

    # drift_score: effect van drift op de gekozen kant
    picks["drift_score"] = np.where(
        picks["PickType"] == "MAIN",
        picks.get("SelectedAdjStrength", np.nan) - picks.get("SelectedRawStrength", np.nan),
        np.nan
    )

    # snapshot_count
    picks["snapshot_count"] = picks["n_snapshots"].fillna(0).astype(int)

    # strength_bucket
    if "SecondaryStrength" not in picks.columns:
        picks["SecondaryStrength"] = np.nan

    strength_for_bucket = np.where(
        picks["PickType"] == "MAIN",
        picks.get("SelectedStrength", picks[strength_col]),
        picks["SecondaryStrength"]
    )
    picks["strength_bucket"] = pd.Series(strength_for_bucket, index=picks.index).apply(make_strength_bucket)

    # drift_range: van gekozen kant
    picks["drift_range"] = np.where(
        picks["Selection"] == "HOME",
        picks["home_range"],
        np.where(
            picks["Selection"] == "AWAY",
            picks["away_range"],
            np.nan
        )
    )

    picks = picks.sort_values(["PickType", "SortStrength"], ascending=[True, False]).copy()

    return picks