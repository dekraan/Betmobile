import pandas as pd

from db import db_engine


def save_model_match_snapshots_to_db(
    df_all: pd.DataFrame,
    picks: pd.DataFrame | None = None,
    run_id=None,
):
    """
    Slaat per run alle wedstrijden op zoals het model ze op dat moment zag.

    Dit is géén odds-history.
    Dit is een modelbeslissing-snapshot:
    - welke stats had de wedstrijd?
    - waarom wel/niet pick?
    - welke fail reasons?
    - was het uiteindelijk een pick?
    - zo ja: welke tier/tags?
    """
    if df_all is None or df_all.empty:
        print("Geen model match snapshots om op te slaan.")
        return

    df = df_all.copy()

    # -------------------------------------------------
    # Pick-info eraan hangen
    # -------------------------------------------------
    if picks is not None and not picks.empty:
        pick_cols = [
            "match_id",
            "PickType",
            "Selection",
            "Advice",
            "pick_tier",
            "pick_stars",
            "sector_tags",
            "danger_tags",
            "classification_reason",
            "selected_odds",
            "selected_prob",
            "selected_value_score",
            "selected_drift_pct",
            "classification_strength",
            "odds_bucket",
            "prob_bucket",
            "rating_gap_bucket",
            "value_bucket",
            "market_support",
            "strength_bucket",
            "snapshot_bucket",
            "drift_bucket",
            "season_phase",
            "weekday",
            "month",
            "passes_danger_combo_v2",
            "passes_danger_combo_v2_no_longshots",
        ]

        pick_cols = [c for c in pick_cols if c in picks.columns]

        pick_info = picks[pick_cols].copy()
        pick_info = pick_info.drop_duplicates(subset=["match_id"], keep="first")

        df = df.merge(
            pick_info,
            on="match_id",
            how="left",
            suffixes=("", "_pick"),
        )

    df["run_id"] = run_id
    df["is_pick"] = df.get("PickType").notna() if "PickType" in df.columns else False

    # -------------------------------------------------
    # Kolommen mappen naar databasekolommen
    # -------------------------------------------------
    rename_map = {
        "Home Prob": "prob_home",
        "Draw Prob": "prob_draw",
        "Away Prob": "prob_away",

        "AwayRule": "away_rule",
        "HomeRule": "home_rule",
        "Rule": "model_rule_passed",

        "RawStrength_Away_All": "raw_strength_away_all",
        "RawStrength_Home_All": "raw_strength_home_all",
        "RawStrength_Away": "raw_strength_away",
        "RawStrength_Home": "raw_strength_home",

        "RuleStrength": "rule_strength",
        "RuleStrengthAdj": "rule_strength_adj",
        "RuleStrengthCalibrated": "rule_strength_calibrated",

        "RuleStrengthAdj_Home": "rule_strength_adj_home",
        "RuleStrengthAdj_Away": "rule_strength_adj_away",
        "RuleStrengthCalibrated_Home": "rule_strength_calibrated_home",
        "RuleStrengthCalibrated_Away": "rule_strength_calibrated_away",

        "DominantStrengthSide": "dominant_strength_side",
        "DriftNotes": "drift_notes",
        "DriftNotes_Home": "drift_notes_home",
        "DriftNotes_Away": "drift_notes_away",

        "PickType": "pick_type",
        "Selection": "selection",
        "Advice": "advice",
    }

    df = df.rename(columns=rename_map)

    wanted_cols = [
        "run_id",

        "match_id",
        "date",
        "competition",
        "home_team",
        "away_team",

        "odds_home",
        "odds_draw",
        "odds_away",

        "prob_home",
        "prob_draw",
        "prob_away",

        "bet_home",
        "bet_draw",
        "bet_away",

        "home_rating",
        "away_rating",
        "rating_gap",
        "rating_home_edge",

        "n_snapshots",
        "hours_stale",
        "market_age_hours",
        "hours_to_kickoff",
        "scrape_to_kickoff_hours",

        "home_drift_pct",
        "away_drift_pct",
        "home_drift_abs",
        "away_drift_abs",
        "home_range",
        "away_range",
        "home_last_move_pct",
        "away_last_move_pct",
        "home_recent24_pct",
        "away_recent24_pct",

        "away_rule",
        "home_rule",
        "model_rule_passed",

        "away_fail_reasons",
        "home_fail_reasons",
        "away_fail_count",
        "home_fail_count",
        "closest_side",
        "closest_fail_reasons",

        "raw_strength_away_all",
        "raw_strength_home_all",
        "raw_strength_away",
        "raw_strength_home",

        "rule_strength",
        "rule_strength_adj",
        "rule_strength_calibrated",

        "rule_strength_adj_home",
        "rule_strength_adj_away",
        "rule_strength_calibrated_home",
        "rule_strength_calibrated_away",

        "dominant_strength_side",
        "drift_notes",
        "drift_notes_home",
        "drift_notes_away",

        "is_pick",
        "pick_type",
        "selection",
        "advice",
        "pick_tier",
        "pick_stars",
        "sector_tags",
        "danger_tags",
        "classification_reason",

        "selected_odds",
        "selected_prob",
        "selected_value_score",
        "selected_drift_pct",
        "classification_strength",

        "odds_bucket",
        "prob_bucket",
        "rating_gap_bucket",
        "value_bucket",
        "market_support",
        "strength_bucket",
        "snapshot_bucket",
        "drift_bucket",
        "season_phase",
        "weekday",
        "month",

        "passes_danger_combo_v2",
        "passes_danger_combo_v2_no_longshots",
    ]

    existing = [c for c in wanted_cols if c in df.columns]
    out = df[existing].copy()

    # Datum netjes houden
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date

    with db_engine().begin() as conn:
        out.to_sql(
            "model_match_snapshots",
            conn,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    print(f"Model match snapshots opgeslagen: {len(out)}")