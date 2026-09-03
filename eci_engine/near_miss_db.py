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

    # RuleStrengthCalibrated is de match-brede max over beide zijden en staat
    # bij een near miss ALTIJD op 0: apply_drift rekent alleen een adj uit als
    # AwayRule of HomeRule waar is, en dat is bij een near miss nooit zo.
    # Neem daarom de _All variant van de gekozen zijde, net zoals run_model.py
    # dat voor single fails al doet.
    side = near_miss.get("NearMissSide")
    if side is not None and {"RawStrength_Home_All", "RawStrength_Away_All"} <= set(near_miss.columns):
        import numpy as np
        df["RuleStrengthCalibrated"] = np.where(
            side.astype(str).str.upper() == "HOME",
            near_miss["RawStrength_Home_All"],
            np.where(
                side.astype(str).str.upper() == "AWAY",
                near_miss["RawStrength_Away_All"],
                np.nan,
            ),
        )
    else:
        print("[near_miss] WAARSCHUWING: RawStrength_*_All ontbreekt; "
              "strength blijft mogelijk 0.")

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