import pandas as pd
from sqlalchemy import text

from db import db_engine


def save_near_misses_to_db(
    near_miss: pd.DataFrame,
    run_id=None,
):
    if near_miss is None or near_miss.empty:
        print("Geen near misses om op te slaan.")
        return

    cols = [
        "match_id",
        "date",
        "competition",
        "home_team",
        "away_team",

        "NearMissSide",
        "NearMissReason",

        "single_fail_margin",

        "prob_margin",
        "value_margin",
        "odds_margin",
        "drift_margin",
        "rating_margin",
        "edge_margin",

        "snap_needed",

        "selected_prob_sf",
        "selected_value_sf",
        "selected_odds_sf",
        "selected_drift_sf",

        "RuleStrengthCalibrated",
    ]

    existing = [c for c in cols if c in near_miss.columns]

    df = near_miss[existing].copy()

    rename_map = {
        "NearMissSide": "side",
        "NearMissReason": "fail_reason",

        "selected_prob_sf": "selected_prob",
        "selected_value_sf": "selected_value",
        "selected_odds_sf": "selected_odds",
        "selected_drift_sf": "selected_drift",

        "RuleStrengthCalibrated": "strength",
    }

    df = df.rename(columns=rename_map)

    df["run_id"] = run_id

    with db_engine().begin() as conn:
        df.to_sql(
            "picks_near_miss_candidates",
            conn,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    print(f"{len(df)} near misses opgeslagen.")