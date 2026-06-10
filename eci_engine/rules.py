import pandas as pd
import numpy as np

from config import (
    RULE_MIN_PROB,
    RULE_MIN_VALUE,
    RULE_MIN_RATING_GAP,
    RULE_MIN_ODDS,
    RULE_MAX_ODDS,
    RULE_MIN_SNAPSHOTS,
    RULE_MIN_DRIFT_ABS,
    DRIFT_OPPOSE_THRESHOLD,
    DRIFT_SUPPORT_THRESHOLD,
    DRIFT_SUPPORT_BONUS,
    DRIFT_OPPOSE_PENALTY,
    SNAP_BONUS_THRESHOLD,
    SNAP_BONUS,
    RANGE_PENALTY_THRESHOLD,
    RANGE_PENALTY,
    STALE_PENALTY_THRESHOLD,
    STALE_PENALTY,
    KICKOFF_SOON_HOURS,
    KICKOFF_SOON_BONUS,
    KICKOFF_VERY_SOON_HOURS,
    KICKOFF_VERY_SOON_BONUS,
    STALE_NEAR_KICKOFF_HOURS,
    STALE_DAY_KICKOFF_HOURS,
    DRIFT_STRONG_THRESHOLD,
    DRIFT_STRONG_BONUS,
    DRIFT_STRONG_PENALTY,
    DRIFT_CONSISTENCY_THRESHOLD,
    DRIFT_CONSISTENCY_BONUS,
    DRIFT_CONSISTENCY_PENALTY,
    DRIFT_NOISE_MULTIPLIER,
    DRIFT_NOISE_PENALTY,
    LAST_MOVE_SUPPORT_THRESHOLD,
    LAST_MOVE_OPPOSE_THRESHOLD,
    LAST_MOVE_BONUS,
    LAST_MOVE_PENALTY,
    RECENT24_SUPPORT_THRESHOLD,
    RECENT24_OPPOSE_THRESHOLD,
    RECENT24_BONUS,
    RECENT24_PENALTY,
)
from utils import safe_float, calc_drift_consistency


# =====================================================================
# APPLY RULES (HOME & AWAY) — v4.1 CLEAN
# =====================================================================
def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------------------
    # 1. Boolean rule checks
    # -----------------------------------------

    # AWAY
    away_prob_ok     = df["Away Prob"] >= RULE_MIN_PROB
    away_value_ok    = df["bet_away"] >= RULE_MIN_VALUE
    away_rating_ok   = df["rating_gap"] >= RULE_MIN_RATING_GAP
    away_odds_ok     = df["odds_away"].between(RULE_MIN_ODDS, RULE_MAX_ODDS)
    away_snap_ok     = df["n_snapshots"].fillna(0) >= RULE_MIN_SNAPSHOTS
    away_drift_size_ok = df["away_drift_pct"].abs().fillna(0) >= RULE_MIN_DRIFT_ABS
    away_drift_support_ok = df["away_drift_pct"].fillna(0) <= DRIFT_OPPOSE_THRESHOLD

    away_final = (
        away_prob_ok &
        away_value_ok &
        away_rating_ok &
        away_odds_ok &
        away_snap_ok &
        away_drift_size_ok &
        away_drift_support_ok
    )

    # HOME
    home_prob_ok     = df["Home Prob"] >= RULE_MIN_PROB
    home_value_ok    = df["bet_home"] >= RULE_MIN_VALUE
    home_rating_ok   = df["rating_gap"] >= RULE_MIN_RATING_GAP
    home_odds_ok     = df["odds_home"].between(RULE_MIN_ODDS, RULE_MAX_ODDS)
    home_snap_ok     = df["n_snapshots"].fillna(0) >= RULE_MIN_SNAPSHOTS
    home_drift_size_ok = df["home_drift_pct"].abs().fillna(0) >= RULE_MIN_DRIFT_ABS
    home_drift_support_ok = df["home_drift_pct"].fillna(0) <= DRIFT_OPPOSE_THRESHOLD
    edge_ok          = df["rating_home_edge"] >= 0

    home_final = (
        home_prob_ok &
        home_value_ok &
        home_rating_ok &
        home_odds_ok &
        home_snap_ok &
        home_drift_size_ok &
        home_drift_support_ok &
        edge_ok
    )

    # Opslaan
    df["AwayRule"] = away_final
    df["HomeRule"] = home_final
    df["Rule"] = away_final | home_final

    # -----------------------------------------
    # 1b. Bewaar rule checks als losse kolommen (nieuw)
    # -----------------------------------------

    # AWAY checks -> kolommen
    df["away_prob_ok"]   = away_prob_ok
    df["away_value_ok"]  = away_value_ok
    df["away_rating_ok"] = away_rating_ok
    df["away_odds_ok"]   = away_odds_ok
    df["away_snap_ok"]   = away_snap_ok
    df["away_drift_size_ok"] = away_drift_size_ok
    df["away_drift_support_ok"] = away_drift_support_ok
    df["away_drift_ok"] = away_drift_size_ok & away_drift_support_ok
    df["away_final"]     = away_final

    # HOME checks -> kolommen
    df["home_prob_ok"]   = home_prob_ok
    df["home_value_ok"]  = home_value_ok
    df["home_rating_ok"] = home_rating_ok
    df["home_odds_ok"]   = home_odds_ok
    df["home_snap_ok"]   = home_snap_ok
    df["home_drift_size_ok"] = home_drift_size_ok
    df["home_drift_support_ok"] = home_drift_support_ok
    df["home_drift_ok"] = home_drift_size_ok & home_drift_support_ok
    df["home_edge_ok"]   = edge_ok
    df["home_final"]     = home_final

    # Extra detail: welk deel van drift faalde?
    df["away_drift_fail_detail"] = np.select(
        [
            (~df["away_drift_size_ok"]) & (~df["away_drift_support_ok"]),
            (~df["away_drift_size_ok"]),
            (~df["away_drift_support_ok"]),
        ],
        [
            "size+support",
            "size",
            "support",
        ],
        default=None
    )

    df["home_drift_fail_detail"] = np.select(
        [
            (~df["home_drift_size_ok"]) & (~df["home_drift_support_ok"]),
            (~df["home_drift_size_ok"]),
            (~df["home_drift_support_ok"]),
        ],
        [
            "size+support",
            "size",
            "support",
        ],
        default=None
    )

    # Fail reasons + fail count (handig voor filteren)
    away_checks = ["prob", "value", "rating", "odds", "snap", "drift"]
    home_checks = ["prob", "value", "rating", "odds", "snap", "drift", "edge"]

    def _fail_list(row, prefix, checks):
        fails = []
        for name in checks:
            col = f"{prefix}_{name}_ok"
            if col in row and (not bool(row[col])):
                fails.append(name)
        return ",".join(fails)

    df["away_fail_reasons"] = df.apply(lambda r: _fail_list(r, "away", away_checks), axis=1)
    df["home_fail_reasons"] = df.apply(lambda r: _fail_list(r, "home", home_checks), axis=1)

    df["away_fail_count"] = df[[
        "away_prob_ok","away_value_ok","away_rating_ok","away_odds_ok","away_snap_ok","away_drift_ok"
    ]].eq(False).sum(axis=1)

    df["home_fail_count"] = df[[
        "home_prob_ok","home_value_ok","home_rating_ok","home_odds_ok","home_snap_ok","home_drift_ok","home_edge_ok"
    ]].eq(False).sum(axis=1)

    # "bijna pick" helpers
    df["away_almost"] = (df["away_fail_count"] == 1) & (~df["away_final"])
    df["home_almost"] = (df["home_fail_count"] == 1) & (~df["home_final"])

    # Welke kant zat het dichtst bij een pick?
    def _closest_side(r):
        if r["away_fail_count"] < r["home_fail_count"]:
            return "AWAY"
        if r["home_fail_count"] < r["away_fail_count"]:
            return "HOME"
        # evenveel fails → geen duidelijke voorkeur
        return None

    df["closest_side"] = df.apply(_closest_side, axis=1)

    # Fail reasons van de kant die het dichtst bij een pick zat
    df["closest_fail_reasons"] = np.where(
        df["closest_side"] == "AWAY",
        df["away_fail_reasons"],
        np.where(
            df["closest_side"] == "HOME",
            df["home_fail_reasons"],
            None
        )
    )


    # -----------------------------------------
    # 2. Build reason text (per rij)
    # -----------------------------------------
    def away_reason(r):
        return (
            f"AWAY:"
            f"prob={safe_float(r['Away Prob'])}|min={RULE_MIN_PROB}|ok={r['away_prob_ok']};"
            f"value={safe_float(r['bet_away'])}|min={RULE_MIN_VALUE}|ok={r['away_value_ok']};"
            f"rating_gap={safe_float(r['rating_gap'])}|min={RULE_MIN_RATING_GAP}|ok={r['away_rating_ok']};"
            f"odds={safe_float(r['odds_away'])}|range=[{RULE_MIN_ODDS},{RULE_MAX_ODDS}]|ok={r['away_odds_ok']};"
            f"snapshots={safe_float(r['n_snapshots'])}|min={RULE_MIN_SNAPSHOTS}|ok={r['away_snap_ok']};"
            f"drift_pct={safe_float(r.get('away_drift_pct'))}|"
            f"size_ok={r['away_drift_size_ok']}|min_abs={RULE_MIN_DRIFT_ABS}|"
            f"support_ok={r['away_drift_support_ok']}|oppose_max={DRIFT_OPPOSE_THRESHOLD}|"
            f"drift_ok={r['away_drift_ok']}|"
            f"drift_fail_detail={r.get('away_drift_fail_detail')};"
            f"final={r['AwayRule']}"
        )

    def home_reason(r):
        return (
            f"HOME:"
            f"prob={safe_float(r['Home Prob'])}|min={RULE_MIN_PROB}|ok={r['home_prob_ok']};"
            f"value={safe_float(r['bet_home'])}|min={RULE_MIN_VALUE}|ok={r['home_value_ok']};"
            f"rating_gap={safe_float(r['rating_gap'])}|min={RULE_MIN_RATING_GAP}|ok={r['home_rating_ok']};"
            f"odds={safe_float(r['odds_home'])}|range=[{RULE_MIN_ODDS},{RULE_MAX_ODDS}]|ok={r['home_odds_ok']};"
            f"snapshots={safe_float(r['n_snapshots'])}|min={RULE_MIN_SNAPSHOTS}|ok={r['home_snap_ok']};"
            f"drift_pct={safe_float(r.get('home_drift_pct'))}|"
            f"size_ok={r['home_drift_size_ok']}|min_abs={RULE_MIN_DRIFT_ABS}|"
            f"support_ok={r['home_drift_support_ok']}|oppose_max={DRIFT_OPPOSE_THRESHOLD}|"
            f"drift_ok={r['home_drift_ok']}|"
            f"drift_fail_detail={r.get('home_drift_fail_detail')};"
            f"edge={safe_float(r['rating_home_edge'])}|min=0|ok={r['home_edge_ok']};"
            f"final={r['HomeRule']}"
        )

    df["Away_reason"] = df.apply(away_reason, axis=1)
    df["Home_reason"] = df.apply(home_reason, axis=1)

    def combine(r):
        if r["AwayRule"] and not r["HomeRule"]:
            return r["Away_reason"]
        if r["HomeRule"] and not r["AwayRule"]:
            return r["Home_reason"]
        return r["Away_reason"] + " | " + r["Home_reason"]

    df["Rule_reason"] = df.apply(combine, axis=1)
    df["rule_passed"] = df["Rule"]

    # -----------------------------------------
    # 3. Strength scoring (ECI v3 → v4 identical)
    # -----------------------------------------

    raw_away = (
        (df["bet_away"] - RULE_MIN_VALUE).clip(lower=0) * 2 +
        (df["Away Prob"] - RULE_MIN_PROB).clip(lower=0) * 2 +
        df["rating_gap"].clip(lower=0) / 500
    )

    raw_home = (
        (df["bet_home"] - RULE_MIN_VALUE).clip(lower=0) * 2 +
        (df["Home Prob"] - RULE_MIN_PROB).clip(lower=0) * 2 +
        df["rating_gap"].clip(lower=0) / 500 +
        df["rating_home_edge"].clip(lower=0) / 500
    )

    # Altijd bewaren, ook als de rule niet gehaald is
    df["RawStrength_Away_All"] = raw_away
    df["RawStrength_Home_All"] = raw_home

    # Alleen voor MAIN picks tellen ze mee in RuleStrength
    df["RawStrength_Away"] = np.where(df["AwayRule"], raw_away, 0)
    df["RawStrength_Home"] = np.where(df["HomeRule"], raw_home, 0)

    df["RuleStrength"] = df[["RawStrength_Away", "RawStrength_Home"]].max(axis=1)

    return df



# =====================================================================
# APPLY DRIFT PENALTY
# =====================================================================
def apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def adjust_one_side(base_strength, side_prefix, row):
        s = float(base_strength)
        notes = []

        d_pct = row.get(f"{side_prefix}_drift_pct")
        snaps = row.get("n_snapshots")
        rng = row.get(f"{side_prefix}_range")
        stale = row.get("hours_stale")
        hours_to_kickoff = row.get("hours_to_kickoff")
        scrape_to_kickoff = row.get("scrape_to_kickoff_hours")
        last_move = row.get(f"{side_prefix}_last_move_pct")
        recent24 = row.get(f"{side_prefix}_recent24_pct")

        consistency = calc_drift_consistency(d_pct, rng)

        # Richting drift
        if pd.notna(d_pct):
            if d_pct <= DRIFT_SUPPORT_THRESHOLD:
                s += DRIFT_SUPPORT_BONUS
                notes.append(f"{side_prefix}_market_support")
            elif d_pct >= DRIFT_OPPOSE_THRESHOLD:
                s -= DRIFT_OPPOSE_PENALTY
                notes.append(f"{side_prefix}_market_oppose")

        # Laatste move
        if pd.notna(last_move):
            if last_move <= LAST_MOVE_SUPPORT_THRESHOLD:
                s += LAST_MOVE_BONUS
                notes.append(f"{side_prefix}_last_move_support")
            elif last_move >= LAST_MOVE_OPPOSE_THRESHOLD:
                s -= LAST_MOVE_PENALTY
                notes.append(f"{side_prefix}_last_move_oppose")

        # Recente 24u move
        if pd.notna(recent24):
            if recent24 <= RECENT24_SUPPORT_THRESHOLD:
                s += RECENT24_BONUS
                notes.append(f"{side_prefix}_recent24_support")
            elif recent24 >= RECENT24_OPPOSE_THRESHOLD:
                s -= RECENT24_PENALTY
                notes.append(f"{side_prefix}_recent24_oppose")

        # Sterke move
        if pd.notna(d_pct):
            if d_pct <= -DRIFT_STRONG_THRESHOLD:
                s += DRIFT_STRONG_BONUS
                notes.append(f"{side_prefix}_strong_support")
            elif d_pct >= DRIFT_STRONG_THRESHOLD:
                s -= DRIFT_STRONG_PENALTY
                notes.append(f"{side_prefix}_strong_oppose")

        # Consistente move
        if consistency is not None:
            if pd.notna(d_pct) and d_pct <= DRIFT_SUPPORT_THRESHOLD and consistency >= DRIFT_CONSISTENCY_THRESHOLD:
                s += DRIFT_CONSISTENCY_BONUS
                notes.append(f"{side_prefix}_consistent_support")
            elif pd.notna(d_pct) and d_pct >= DRIFT_OPPOSE_THRESHOLD and consistency >= DRIFT_CONSISTENCY_THRESHOLD:
                s -= DRIFT_CONSISTENCY_PENALTY
                notes.append(f"{side_prefix}_consistent_oppose")

        # Veel snapshots
        if pd.notna(snaps) and snaps >= SNAP_BONUS_THRESHOLD:
            s += SNAP_BONUS
            notes.append(f"{side_prefix}_snap_bonus")

        # Grote range
        if pd.notna(rng) and rng >= RANGE_PENALTY_THRESHOLD:
            s -= RANGE_PENALTY
            notes.append(f"{side_prefix}_range_penalty")

        # Noisy drift
        if pd.notna(rng) and pd.notna(d_pct):
            if rng >= max(RANGE_PENALTY_THRESHOLD, abs(d_pct) * DRIFT_NOISE_MULTIPLIER):
                s -= DRIFT_NOISE_PENALTY
                notes.append(f"{side_prefix}_noisy_drift")

        # Bonus als marktsteun laat is
        if pd.notna(d_pct) and pd.notna(scrape_to_kickoff):
            if d_pct <= DRIFT_SUPPORT_THRESHOLD and scrape_to_kickoff <= KICKOFF_SOON_HOURS:
                s += KICKOFF_SOON_BONUS
                notes.append(f"{side_prefix}_support_near_kickoff")
            if d_pct <= DRIFT_SUPPORT_THRESHOLD and scrape_to_kickoff <= KICKOFF_VERY_SOON_HOURS:
                s += KICKOFF_VERY_SOON_BONUS
                notes.append(f"{side_prefix}_support_very_near_kickoff")
        elif pd.notna(d_pct) and pd.notna(hours_to_kickoff):
            if d_pct <= DRIFT_SUPPORT_THRESHOLD and hours_to_kickoff <= KICKOFF_SOON_HOURS:
                s += KICKOFF_SOON_BONUS
                notes.append(f"{side_prefix}_support_near_kickoff")
            if d_pct <= DRIFT_SUPPORT_THRESHOLD and hours_to_kickoff <= KICKOFF_VERY_SOON_HOURS:
                s += KICKOFF_VERY_SOON_BONUS
                notes.append(f"{side_prefix}_support_very_near_kickoff")

        # Stale penalty
        if pd.notna(stale) and pd.notna(hours_to_kickoff):
            if hours_to_kickoff <= 6 and stale >= STALE_NEAR_KICKOFF_HOURS:
                s -= STALE_PENALTY
                notes.append(f"{side_prefix}_stale_close_to_kickoff")
            elif hours_to_kickoff <= 24 and stale >= STALE_DAY_KICKOFF_HOURS:
                s -= STALE_PENALTY
                notes.append(f"{side_prefix}_stale_day_before_kickoff")
            elif stale >= STALE_PENALTY_THRESHOLD:
                s -= STALE_PENALTY
                notes.append(f"{side_prefix}_stale_penalty")
        elif pd.notna(stale) and stale >= STALE_PENALTY_THRESHOLD:
            s -= STALE_PENALTY
            notes.append(f"{side_prefix}_stale_penalty")

        return max(0.0, s), notes

    def adjust(row):
        away_adj = 0.0
        home_adj = 0.0
        away_notes = []
        home_notes = []

        if bool(row.get("AwayRule")):
            away_adj, away_notes = adjust_one_side(row["RawStrength_Away"], "away", row)

        if bool(row.get("HomeRule")):
            home_adj, home_notes = adjust_one_side(row["RawStrength_Home"], "home", row)

        final_adj = max(away_adj, home_adj)

        if away_adj > home_adj:
            dominant_side = "AWAY"
            dominant_notes = away_notes
        elif home_adj > away_adj:
            dominant_side = "HOME"
            dominant_notes = home_notes
        elif away_adj > 0 and home_adj > 0:
            dominant_side = "AWAY" if row["RawStrength_Away"] >= row["RawStrength_Home"] else "HOME"
            dominant_notes = away_notes if dominant_side == "AWAY" else home_notes
        else:
            dominant_side = None
            dominant_notes = []

        return pd.Series({
            "RuleStrengthAdj_Away": away_adj,
            "RuleStrengthAdj_Home": home_adj,
            "RuleStrengthAdj": final_adj,
            "DominantStrengthSide": dominant_side,
            "DriftNotes_Away": ",".join(away_notes) if away_notes else None,
            "DriftNotes_Home": ",".join(home_notes) if home_notes else None,
            "DriftNotes": ",".join(dominant_notes) if dominant_notes else None
        })

    adjusted = df.apply(adjust, axis=1)
    for col in adjusted.columns:
        df[col] = adjusted[col]

    return df
