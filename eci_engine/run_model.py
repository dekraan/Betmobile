import pandas as pd
import numpy as np

from db import auto_exclude_past_no_score_matches, refresh_source_views
from data_loader import load_upcoming, load_calibration_history
from rules import apply_rules, apply_drift
from calibration import build_calibration, apply_calibration
from picks import build_picks
from classification import classify_picks
from reporting import analyze_failures
from exporters import save_to_db, save_excel, save_single_fails_to_db
from utils import choose_relevant_side
from single_fail_features import enrich_single_fail_features
from model_snapshot_db import save_model_match_snapshots_to_db
from near_miss_db import save_near_misses_to_db
from config import (
    MIN_STRENGTH,
    RULE_MIN_SNAPSHOTS,
    RULE_MIN_VALUE,
    RULE_MIN_ODDS,
    RULE_MAX_ODDS,
    RULE_MIN_PROB,
    RULE_MIN_RATING_GAP,
    RULE_MIN_DRIFT_ABS,
    DRIFT_OPPOSE_THRESHOLD,
)

# =====================================================================
# MAIN
# =====================================================================
def main():
    print("\n=== ECI PICK ENGINE v4.1 RUN ===")

    # 0. Oude ECI-wedstrijden zonder score automatisch uitsluiten
    auto_exclude_past_no_score_matches()

    # 1. Daarna de onderliggende views verversen
    refresh_source_views()

    # 2. Upcoming matches
    df = load_upcoming()
    if df.empty:
        print("[load_upcoming] Geen upcoming data.")
        return

    # 3. Drift features zitten nu al in betmobile via load_upcoming()

    # 4. Historische calibratie
    hist = load_calibration_history()
    calib_meta, calib_table = build_calibration(hist)

    # 5. Rules + drift + calibratie
    df = apply_rules(df)
    analyze_failures(df)
    df = apply_drift(df)
    df = apply_calibration(df, calib_table)

    # 5a. Bouw MAIN + SECONDARY picks
    picks = build_picks(df)
    picks = classify_picks(picks)
    # 5b) WATCHLIST: alleen snap faalt (dus inhoudelijk OK, maar nog niet genoeg gescraped)
    snap_watch = df[
        (
            (df["home_fail_count"] == 1) & (df["home_fail_reasons"] == "snap")
        ) | (
            (df["away_fail_count"] == 1) & (df["away_fail_reasons"] == "snap")
        )
    ].copy()

    if not snap_watch.empty:
        snap_watch["WatchSide"] = snap_watch.apply(
            lambda r: choose_relevant_side(
                r,
                home_condition=((r["home_fail_count"] == 1) and (r["home_fail_reasons"] == "snap")),
                away_condition=((r["away_fail_count"] == 1) and (r["away_fail_reasons"] == "snap")),
            ),
            axis=1
        )
        snap_watch = snap_watch[snap_watch["WatchSide"].notna()].copy()
        # hoeveel snapshots nog nodig?
        snap_watch["snap_needed"] = (RULE_MIN_SNAPSHOTS - snap_watch["n_snapshots"].fillna(0)).clip(lower=0)

        strength_col = "RuleStrengthCalibrated" if "RuleStrengthCalibrated" in df.columns else "RuleStrengthAdj"
        snap_watch = snap_watch.sort_values([strength_col, "snap_needed"], ascending=[False, True])

    # 5c) WATCHLIST: alleen odds of alleen value faalt (discussiegevallen)
    price_watch = df[
        (
            (df["home_fail_count"] == 1) & (df["home_fail_reasons"].isin(["odds", "value"]))
        ) | (
            (df["away_fail_count"] == 1) & (df["away_fail_reasons"].isin(["odds", "value"]))
        )
    ].copy()

    if not price_watch.empty:
        price_watch["WatchSide"] = price_watch.apply(
            lambda r: choose_relevant_side(
                r,
                home_condition=((r["home_fail_count"] == 1) and (r["home_fail_reasons"] in ["odds", "value"])),
                away_condition=((r["away_fail_count"] == 1) and (r["away_fail_reasons"] in ["odds", "value"])),
            ),
            axis=1
        )
        price_watch = price_watch[price_watch["WatchSide"].notna()].copy()

        price_watch["WatchReason"] = np.where(
            price_watch["WatchSide"] == "HOME",
            price_watch["home_fail_reasons"],
            price_watch["away_fail_reasons"]
        )

        # Marges: hoe ver zit odds/value van de grens?
        price_watch["value_margin"] = np.where(
            price_watch["WatchSide"] == "HOME",
            price_watch["bet_home"] - RULE_MIN_VALUE,
            price_watch["bet_away"] - RULE_MIN_VALUE
        )

        price_watch["odds_margin"] = np.where(
            price_watch["WatchSide"] == "HOME",
            np.minimum(
                price_watch["odds_home"] - RULE_MIN_ODDS,
                RULE_MAX_ODDS - price_watch["odds_home"]
            ),
            np.minimum(
                price_watch["odds_away"] - RULE_MIN_ODDS,
                RULE_MAX_ODDS - price_watch["odds_away"]
            )
        )

        strength_col = "RuleStrengthCalibrated" if "RuleStrengthCalibrated" in df.columns else "RuleStrengthAdj"
        price_watch = price_watch.sort_values([strength_col, "value_margin"], ascending=[False, False])

    # 5d) NEAR MISS: alle single-fail kandidaten, ongeacht datum
    near_miss = df[
        (df["home_fail_count"] == 1) | (df["away_fail_count"] == 1)
    ].copy()

    if not near_miss.empty:
        near_miss["NearMissSide"] = near_miss.apply(
            lambda r: choose_relevant_side(
                r,
                home_condition=(r["home_fail_count"] == 1),
                away_condition=(r["away_fail_count"] == 1),
            ),
            axis=1,
        )

        near_miss = near_miss[near_miss["NearMissSide"].notna()].copy()

        near_miss["NearMissReason"] = np.where(
            near_miss["NearMissSide"] == "HOME",
            near_miss["home_fail_reasons"],
            near_miss["away_fail_reasons"],
        )

        near_miss = enrich_single_fail_features(
            near_miss,
            side_col="NearMissSide",
            reason_col="NearMissReason",
        )

        strength_col_nm = "RuleStrengthCalibrated" if "RuleStrengthCalibrated" in near_miss.columns else "RuleStrengthAdj"

        near_miss = near_miss.sort_values(
            ["single_fail_margin", strength_col_nm],
            ascending=[False, False],
        )
        near_miss["NearMissRank"] = range(1, len(near_miss) + 1)

    # 5e) TODAY view: alles voor de dag van vandaag
    today = pd.Timestamp.today().normalize()
    df_today = df[pd.to_datetime(df["date"], errors="coerce").dt.normalize() == today].copy()

    # Single-fail voor vandaag: toon alleen de relevante kant
    # - als slechts 1 kant single-fail is -> die kant
    # - als beide kanten single-fail zijn -> kies de kant met hoogste ruwe strength
    single_fail_today = df_today[
        (df_today["home_fail_count"] == 1) | (df_today["away_fail_count"] == 1)
    ].copy()

    single_fail_today["SingleFailSide"] = None
    single_fail_today["SingleFailReason"] = None

    if not single_fail_today.empty:
        single_fail_today["SingleFailSide"] = single_fail_today.apply(
            lambda r: choose_relevant_side(
                r,
                home_condition=(r["home_fail_count"] == 1),
                away_condition=(r["away_fail_count"] == 1),
            ),
            axis=1
        )

        single_fail_today = single_fail_today[
            single_fail_today["SingleFailSide"].notna()
        ].copy()

        single_fail_today["SingleFailReason"] = np.where(
            single_fail_today["SingleFailSide"] == "HOME",
            single_fail_today["home_fail_reasons"],
            np.where(
                single_fail_today["SingleFailSide"] == "AWAY",
                single_fail_today["away_fail_reasons"],
                None
            )
        )
        
        # Strength voor de gekozen single-fail kant
        single_fail_today["SingleFailRawStrength"] = np.where(
            single_fail_today["SingleFailSide"] == "HOME",
            single_fail_today["RawStrength_Home_All"],
            single_fail_today["RawStrength_Away_All"]
        )

        single_fail_today["SingleFailAdjStrength"] = np.where(
            single_fail_today["SingleFailSide"] == "HOME",
            single_fail_today.get("RuleStrengthAdj_Home", single_fail_today["RawStrength_Home_All"]),
            single_fail_today.get("RuleStrengthAdj_Away", single_fail_today["RawStrength_Away_All"])
        )

        single_fail_today["SingleFailCalibratedStrength"] = np.where(
            single_fail_today["SingleFailSide"] == "HOME",
            single_fail_today.get("RuleStrengthCalibrated_Home", single_fail_today["SingleFailAdjStrength"]),
            single_fail_today.get("RuleStrengthCalibrated_Away", single_fail_today["SingleFailAdjStrength"])
        )
        

    # --- Mini verbetering: single_fail_margin (hoe dichtbij de grens?) ---
    single_fail_today["snap_needed"] = (RULE_MIN_SNAPSHOTS - single_fail_today["n_snapshots"].fillna(0)).clip(lower=0)

    sf_side = single_fail_today["SingleFailSide"]
    sf_reason = single_fail_today["SingleFailReason"]

    sf_prob  = np.where(sf_side == "HOME", single_fail_today["Home Prob"], single_fail_today["Away Prob"])
    sf_value = np.where(sf_side == "HOME", single_fail_today["bet_home"],   single_fail_today["bet_away"])
    sf_odds  = np.where(sf_side == "HOME", single_fail_today["odds_home"],  single_fail_today["odds_away"])
    sf_drift = np.where(sf_side == "HOME", single_fail_today["home_drift_pct"], single_fail_today["away_drift_pct"])

    single_fail_today["prob_margin"]  = sf_prob - RULE_MIN_PROB
    single_fail_today["value_margin"] = sf_value - RULE_MIN_VALUE

    single_fail_today["odds_margin"] = np.where(
        sf_odds < RULE_MIN_ODDS,
        sf_odds - RULE_MIN_ODDS,
        np.where(sf_odds > RULE_MAX_ODDS, RULE_MAX_ODDS - sf_odds, 0.0)
    )

    drift_size_margin = np.abs(sf_drift) - RULE_MIN_DRIFT_ABS
    drift_support_margin = DRIFT_OPPOSE_THRESHOLD - sf_drift

    single_fail_today["drift_margin"] = np.where(
        sf_side == "HOME",
        np.where(
            single_fail_today["home_drift_fail_detail"] == "size",
            drift_size_margin,
            np.where(
                single_fail_today["home_drift_fail_detail"] == "support",
                drift_support_margin,
                np.minimum(drift_size_margin, drift_support_margin)
            )
        ),
        np.where(
            single_fail_today["away_drift_fail_detail"] == "size",
            drift_size_margin,
            np.where(
                single_fail_today["away_drift_fail_detail"] == "support",
                drift_support_margin,
                np.minimum(drift_size_margin, drift_support_margin)
            )
        )
    )
    single_fail_today["rating_margin"] = single_fail_today["rating_gap"] - RULE_MIN_RATING_GAP
    single_fail_today["edge_margin"]   = single_fail_today["rating_home_edge"]

    single_fail_today["single_fail_margin"] = np.select(
        [
            sf_reason == "snap",
            sf_reason == "value",
            sf_reason == "odds",
            sf_reason == "prob",
            sf_reason == "drift",
            sf_reason == "rating",
            sf_reason == "edge",
        ],
        [
            -single_fail_today["snap_needed"].astype(float),
            single_fail_today["value_margin"].astype(float),
            single_fail_today["odds_margin"].astype(float),
            single_fail_today["prob_margin"].astype(float),
            single_fail_today["drift_margin"].astype(float),
            single_fail_today["rating_margin"].astype(float),
            single_fail_today["edge_margin"].astype(float),
        ],
        default=np.nan
    )

    # Echte picks voor vandaag (MAIN + eventueel SECONDARY)
    strength_col = "RuleStrengthCalibrated" if "RuleStrengthCalibrated" in df.columns else "RuleStrengthAdj"

    if picks.empty:
        picks_today = pd.DataFrame()
    else:
        picks_today = picks[
            pd.to_datetime(picks["date"], errors="coerce").dt.normalize() == today
        ].copy()

    picks_today["Bucket"] = "PICK"
    single_fail_today["Bucket"] = "SINGLE_FAIL"

    today_view = pd.concat([picks_today, single_fail_today], ignore_index=True)

    # sorteren
    sort_cols = ["Bucket", strength_col]
    asc = [True, False]
    if "single_fail_margin" in today_view.columns:
        sort_cols.append("single_fail_margin")
        asc.append(False)

    today_view["single_fail_margin"] = pd.to_numeric(today_view.get("single_fail_margin"), errors="coerce").fillna(-9999)
    today_view = today_view.sort_values(sort_cols, ascending=asc)

    # Zorg dat de kolommen altijd bestaan, ook als er geen picks zijn
    if "PickType" not in today_view.columns:
        today_view["PickType"] = None
    if "Selection" not in today_view.columns:
        today_view["Selection"] = None
    if "Advice" not in today_view.columns:
        today_view["Advice"] = None

    if not picks.empty and not today_view.empty:
        today_view = today_view.merge(
            picks[["match_id", "PickType", "Selection", "Advice"]],
            on="match_id",
            how="left",
            suffixes=("", "_pick")
        )

        # Als door merge _pick-kolommen ontstaan, vul de hoofd-kolommen daarmee
        if "Selection_pick" in today_view.columns:
            today_view["Selection"] = today_view["Selection_pick"].combine_first(today_view["Selection"])
            today_view = today_view.drop(columns=["Selection_pick"])

        if "Advice_pick" in today_view.columns:
            today_view["Advice"] = today_view["Advice_pick"].combine_first(today_view["Advice"])
            today_view = today_view.drop(columns=["Advice_pick"])

        # Merge kan order verstoren → opnieuw sorteren
        today_view = today_view.sort_values(sort_cols, ascending=asc)

    today_view["TodaySide"] = np.where(
        today_view["Bucket"] == "SINGLE_FAIL",
        today_view["SingleFailSide"],
        np.where(
            today_view["Bucket"] == "PICK",
            np.where(today_view["Selection"].notna(), today_view["Selection"], None),
            None
        )
    )

    today_view["TodayReason"] = np.where(
        today_view["Bucket"] == "SINGLE_FAIL",
        today_view["SingleFailReason"],
        np.where(
            today_view.get("PickType").eq("SECONDARY"),
            "SECONDARY_VALUE",
            "ALL_TRUE"
        )
    )

    print(f"Aantal picks na MIN_STRENGTH={MIN_STRENGTH}: {len(picks)}")
    if not picks.empty:
        cols_show = [
            "match_id",
            "competition",
            "home_team",
            "away_team",
            "PickType",
            "Advice",
            "Selection",
            "odds_home",
            "odds_away",
            "Home Prob",
            "Away Prob",
            "bet_home",
            "bet_away",
            "home_drift_pct",
            "away_drift_pct",
            "n_snapshots",
            "home_range",
            "away_range",
            "hours_stale",
            "market_age_hours",
            "home_last_move_pct",
            "away_last_move_pct",
            "home_recent24_pct",
            "away_recent24_pct",
            "hours_to_kickoff",
            "scrape_to_kickoff_hours",
            "RuleStrength",
            "RuleStrengthAdj",
            "RuleStrengthAdj_Home",
            "RuleStrengthAdj_Away",
            "DominantStrengthSide",
            "SelectedRawStrength",
            "SelectedAdjStrength",
            "SelectedStrength",
            "SecondaryStrength",
            "DriftNotes",
        ]
        if "RuleStrengthCalibrated" in picks.columns:
            cols_show.append("RuleStrengthCalibrated")
        if "RuleStrengthCalibrated_Home" in picks.columns:
            cols_show.append("RuleStrengthCalibrated_Home")
        if "RuleStrengthCalibrated_Away" in picks.columns:
            cols_show.append("RuleStrengthCalibrated_Away")

        cols_show = [c for c in cols_show if c in picks.columns]
        print(picks[cols_show].to_string(index=False))
        
    # 6. Opslaan
    run_id = save_to_db(picks, calib_meta)
    save_model_match_snapshots_to_db(df_all=df, picks=picks, run_id=run_id)
    save_single_fails_to_db(run_id, today_view)
    save_excel(picks, df_all=df, snap_watch=snap_watch, price_watch=price_watch, today_view=today_view, near_miss=near_miss)
    save_near_misses_to_db(near_miss, run_id=run_id)
    print("=== RUN COMPLETE ===")


if __name__ == "__main__":
    main()