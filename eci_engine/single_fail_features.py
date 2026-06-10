import numpy as np
import pandas as pd

from config import (
    RULE_MIN_PROB,
    RULE_MIN_VALUE,
    RULE_MIN_RATING_GAP,
    RULE_MIN_ODDS,
    RULE_MAX_ODDS,
    RULE_MIN_SNAPSHOTS,
    RULE_MIN_DRIFT_ABS,
    DRIFT_OPPOSE_THRESHOLD,
)
from utils import choose_relevant_side


def enrich_single_fail_features(df: pd.DataFrame, side_col: str = "SingleFailSide", reason_col: str = "SingleFailReason") -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        return df

    side = df[side_col]
    reason = df[reason_col]

    prob = np.where(side == "HOME", df["Home Prob"], df["Away Prob"])
    value = np.where(side == "HOME", df["bet_home"], df["bet_away"])
    odds = np.where(side == "HOME", df["odds_home"], df["odds_away"])
    drift = np.where(side == "HOME", df["home_drift_pct"], df["away_drift_pct"])

    df["selected_prob_sf"] = prob
    df["selected_value_sf"] = value
    df["selected_odds_sf"] = odds
    df["selected_drift_sf"] = drift

    df["prob_margin"] = prob - RULE_MIN_PROB
    df["value_margin"] = value - RULE_MIN_VALUE

    df["odds_margin"] = np.where(
        odds < RULE_MIN_ODDS,
        odds - RULE_MIN_ODDS,
        np.where(odds > RULE_MAX_ODDS, RULE_MAX_ODDS - odds, 0.0)
    )

    df["snap_needed"] = (RULE_MIN_SNAPSHOTS - df["n_snapshots"].fillna(0)).clip(lower=0)

    drift_size_margin = np.abs(drift) - RULE_MIN_DRIFT_ABS
    drift_support_margin = DRIFT_OPPOSE_THRESHOLD - drift

    home_detail = df.get("home_drift_fail_detail")
    away_detail = df.get("away_drift_fail_detail")

    df["drift_margin"] = np.where(
        side == "HOME",
        np.where(
            home_detail == "size",
            drift_size_margin,
            np.where(home_detail == "support", drift_support_margin, np.minimum(drift_size_margin, drift_support_margin)),
        ),
        np.where(
            away_detail == "size",
            drift_size_margin,
            np.where(away_detail == "support", drift_support_margin, np.minimum(drift_size_margin, drift_support_margin)),
        )
    )

    df["rating_margin"] = df["rating_gap"] - RULE_MIN_RATING_GAP
    df["edge_margin"] = df["rating_home_edge"]

    df["single_fail_margin"] = np.select(
        [
            reason == "snap",
            reason == "value",
            reason == "odds",
            reason == "prob",
            reason == "drift",
            reason == "rating",
            reason == "edge",
        ],
        [
            -df["snap_needed"].astype(float),
            df["value_margin"].astype(float),
            df["odds_margin"].astype(float),
            df["prob_margin"].astype(float),
            df["drift_margin"].astype(float),
            df["rating_margin"].astype(float),
            df["edge_margin"].astype(float),
        ],
        default=np.nan,
    )

    return df