import pandas as pd
import numpy as np
from db import db_engine, get_upcoming_source_name

# =====================================================================
# LOAD UPCOMING MATCHES (uit betmobile)
# =====================================================================
def load_upcoming():
    source_name = get_upcoming_source_name()
    print(f"[load_upcoming] bron = {source_name}")

    q = f"""
        SELECT
            match_id, date, competition,
            home_team, away_team,
            odds_home, odds_draw, odds_away,
            home_win_pct, draw_pct, away_win_pct,
            home_rating, away_rating,
            score,

            home_drift_pct,
            away_drift_pct,
            home_drift_abs,
            away_drift_abs,
            n_snapshots,
            home_range,
            away_range,
            hours_stale,
            market_age_hours,
            home_last_move_pct,
            away_last_move_pct,
            home_recent24_pct,
            away_recent24_pct,
            kickoff_at,
            hours_to_kickoff,
            scrape_to_kickoff_hours
        FROM public.{source_name}
        WHERE score IS NULL
          AND odds_home IS NOT NULL
          AND to_date(date, 'YYYY-MM-DD') >= CURRENT_DATE
        ORDER BY to_date(date, 'YYYY-MM-DD');
    """

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    if df.empty:
        return df

    # Normaliseer percentages -> fractions
    for c in ["home_win_pct", "draw_pct", "away_win_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].max() > 1.2:
            df[c] = df[c] / 100.0

    df["Home Prob"] = df["home_win_pct"]
    df["Draw Prob"] = df["draw_pct"]
    df["Away Prob"] = df["away_win_pct"]

    # Expected value
    df["bet_home"] = df["odds_home"] * df["Home Prob"]
    df["bet_draw"] = df["odds_draw"] * df["Draw Prob"]
    df["bet_away"] = df["odds_away"] * df["Away Prob"]

    df["home_rating"] = pd.to_numeric(df["home_rating"], errors="coerce")
    df["away_rating"] = pd.to_numeric(df["away_rating"], errors="coerce")

    # Rating gap
    df["rating_gap"] = (df["home_rating"] - df["away_rating"]).abs()
    df["rating_home_edge"] = df["home_rating"] - df["away_rating"]

    return df
    
    
    
# =====================================================================
# HISTORICAL CALIBRATION (unmodified but cleaned)
# =====================================================================
def load_calibration_history():
    q = """
        SELECT
            pe.rule_strength_adj,
            pe.outcome,
            pe.selection,
            pe.date_ts,
            CASE
                WHEN pe.selection = 'HOME' THEN pe.odds_home
                WHEN pe.selection = 'AWAY' THEN pe.odds_away
                WHEN pe.selection = 'DRAW' THEN pe.odds_draw
                ELSE NULL
            END AS sel_odds
        FROM public.picks_evaluated pe
        WHERE pe.rule_passed IS TRUE
          AND pe.rule_strength_adj IS NOT NULL
          AND pe.outcome IN ('WIN','LOSS')
    """

    with db_engine().connect() as conn:
        hist = pd.read_sql(q, conn)

    if hist.empty:
        return hist

    hist["profit"] = np.where(hist["outcome"] == "WIN", hist["sel_odds"] - 1, -1)
    return hist