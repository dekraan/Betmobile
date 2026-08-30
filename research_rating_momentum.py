"""
research_rating_momentum.py

Hypothesis: Teams met sterke ECI rating groei (intra-seizoen) outperformen
hun huidige odds, omdat de markt traag reprices op rating-veranderingen.

Doel:
- Snapshot history doorloopen
- Rating delta per team per week meten
- Matchen aan fixtures en odds
- Backtest: CLV edge voor "raters" vs "stale market"

Usage:
    python research_rating_momentum.py
    python research_rating_momentum.py --mode full  # Verbose
    python research_rating_momentum.py --export-csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

# Imports from Betmobile stack
config_loaded = False

# Try: config.py (from main Betmobile stack)
try:
    from config import OUTPUT_DIR
    from db import db_engine
    print("[config] Loaded from config.py & db.py ✓")
    config_loaded = True
except ImportError:
    pass

# Try: betmobile_settings.py (uses .env)
if not config_loaded:
    try:
        from betmobile_settings import DB_DSN
        from sqlalchemy import create_engine
        
        OUTPUT_DIR = Path.cwd() / "output"
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        def db_engine():
            return create_engine(DB_DSN, echo=False)
        
        print("[config] Loaded from betmobile_settings.py (via .env) ✓")
        config_loaded = True
    except Exception as e:
        print(f"[ERROR] Could not load from betmobile_settings.py: {e}")

# Fallback: error
if not config_loaded:
    print("[ERROR] Could not load database config!")
    print("[ERROR] Expected one of:")
    print("  - config.py + db.py in same directory")
    print("  - betmobile_settings.py in same directory (loads from .env)")
    print("[ERROR] Check that your .env has DB_PASSWORD set")
    raise ImportError("No database configuration found. See above.")


# =====================================================================
# CONFIG
# =====================================================================

RATING_DELTA_THRESHOLD = 150  # Mininum ECI point change to flag as "mover"
RATING_LOOKBACK_DAYS = 7     # How far back to measure delta (will be auto-adjusted if not enough history)

# Odds staleness: adjusted if snapshots are very recent
ODDS_STALENESS_MIN = 0       # Odds can be fresh (0 days) if snapshots just started
ODDS_STALENESS_MAX = 365     # But not older than this

MIN_ODDS = 1.30
MAX_ODDS = 3.50
BACKTEST_START_DATE = "2024-09-01"  # Backwards testing period (fallback if many snapshots exist)

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RatingDelta:
    """Single team rating change measurement"""
    team_id: int
    team_name: str
    competition: str
    date_old: datetime
    date_new: datetime
    rating_old: float
    rating_new: float
    delta: float
    direction: str  # "up", "down"


@dataclass
class MatchWithMover:
    """Match where one or both teams are "movers" """
    match_id: int
    date: datetime
    home_team: str
    away_team: str
    competition: str
    
    home_is_mover: bool
    home_delta: float
    away_is_mover: bool
    away_delta: float
    
    home_odds: float | None
    away_odds: float | None
    draw_odds: float | None
    odds_age_days: int
    
    result: str | None  # "home_win", "away_win", "draw", None
    selected_side: str | None  # "home", "away"
    selected_odds: float | None
    won: bool | None


# =====================================================================
# SNAPSHOT LOADING & DELTA CALCULATION
# =====================================================================

def load_eci_snapshots() -> pd.DataFrame:
    """
    Laad match-level snapshots uit eci_data_snapshots.
    Kolommen: snapshot_at, match_id, date, home_team, away_team, 
              home_rating, away_rating, competition
    """
    query = text("""
        SELECT 
            snapshot_at,
            match_id,
            date,
            home_team,
            away_team,
            home_rating,
            away_rating,
            competition
        FROM public.eci_data_snapshots
        WHERE snapshot_at::date >= :start_date
        ORDER BY snapshot_at, match_id
    """)
    
    try:
        with db_engine().connect() as conn:
            df = pd.read_sql(query, conn, params={"start_date": BACKTEST_START_DATE})
    except Exception as e:
        print(f"[ERROR] Could not load snapshots: {e}")
        return pd.DataFrame()
    
    if df.empty:
        print("[WARNING] Snapshots table is empty.")
        return df
    
    # Rename for consistency
    df = df.rename(columns={"snapshot_at": "snapshot_date"})
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["date"] = pd.to_datetime(df["date"])
    
    # Convert ratings to numeric (they may come as strings)
    df["home_rating"] = pd.to_numeric(df["home_rating"], errors="coerce")
    df["away_rating"] = pd.to_numeric(df["away_rating"], errors="coerce")
    
    print(f"[snapshots] Geladen: {len(df)} match-snapshots van {df['snapshot_date'].min()} tot {df['snapshot_date'].max()}")
    print(f"[snapshots] Unique matches: {df['match_id'].nunique()}")
    return df


def calculate_rating_deltas(snapshots: pd.DataFrame, lookback_days: int = 7) -> list[RatingDelta]:
    """
    Bereken rating verandering per team uit match-level snapshots.
    
    Logic:
    - "Unpivot" each match snapshot into home_team & away_team rows
    - Track per team: welke ratings zie je voor dat team over tijd?
    - Bereken delta tussen snapshots
    - Flag als "mover" als delta > threshold
    """
    deltas = []
    
    if snapshots.empty:
        return deltas
    
    snapshots = snapshots.copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])
    
    # Check span of data
    date_min = snapshots["snapshot_date"].min()
    date_max = snapshots["snapshot_date"].max()
    span_days = (date_max - date_min).days
    
    effective_lookback = min(lookback_days, max(1, span_days))
    
    if effective_lookback < lookback_days:
        print(f"[deltas] Data only spans {span_days} days (< {lookback_days}). Using {effective_lookback} day lookback instead.")
    
    # Unpivot: home team + away team per snapshot
    team_ratings = []
    
    for _, row in snapshots.iterrows():
        home_rating = pd.to_numeric(row["home_rating"], errors="coerce")
        away_rating = pd.to_numeric(row["away_rating"], errors="coerce")
        
        if pd.notna(home_rating):
            team_ratings.append({
                "snapshot_date": row["snapshot_date"],
                "team": row["home_team"],
                "competition": row["competition"],
                "rating": home_rating,
                "match_id": row["match_id"],
            })
        
        if pd.notna(away_rating):
            team_ratings.append({
                "snapshot_date": row["snapshot_date"],
                "team": row["away_team"],
                "competition": row["competition"],
                "rating": away_rating,
                "match_id": row["match_id"],
            })
    
    ratings_df = pd.DataFrame(team_ratings)
    
    # For each (team, competition), calculate deltas
    for (team, competition), group in ratings_df.groupby(["team", "competition"]):
        group = group.sort_values("snapshot_date").reset_index(drop=True)
        
        if group.empty or len(group) < 2:
            continue
        
        # For each snapshot, find the rating from ~effective_lookback days ago
        for idx in range(len(group)):
            date_new = group.loc[idx, "snapshot_date"]
            rating_new = group.loc[idx, "rating"]
            
            # Find snapshot from ~effective_lookback days back
            target_date = date_new - timedelta(days=effective_lookback)
            earlier = group[group["snapshot_date"] <= target_date]
            
            if earlier.empty:
                continue
            
            # Take closest earlier snapshot
            earlier_idx = earlier.index[-1]
            date_old = group.loc[earlier_idx, "snapshot_date"]
            rating_old = group.loc[earlier_idx, "rating"]
            
            delta = rating_new - rating_old
            
            # Only register if significant
            if abs(delta) > 50:
                deltas.append(
                    RatingDelta(
                        team_id=0,  # Not used for match-level snapshots
                        team_name=team,
                        competition=competition,
                        date_old=date_old,
                        date_new=date_new,
                        rating_old=rating_old,
                        rating_new=rating_new,
                        delta=delta,
                        direction="up" if delta > 0 else "down",
                    )
                )
    
    return deltas


def identify_movers(deltas: list[RatingDelta], threshold: float = 150) -> pd.DataFrame:
    """Convert deltas naar dataframe, filter major movers"""
    df = pd.DataFrame([
        {
            "team_id": d.team_id,
            "team_name": d.team_name,
            "competition": d.competition,
            "date_delta": d.date_new,
            "rating_old": d.rating_old,
            "rating_new": d.rating_new,
            "delta": d.delta,
            "direction": d.direction,
        }
        for d in deltas
    ])
    
    # Filter naar major movers
    df_movers = df[abs(df["delta"]) >= threshold].copy()
    print(f"[movers] Identified {len(df_movers)} major movers (delta >= ±{threshold})")
    
    return df_movers


# =====================================================================
# MATCH & ODDS LOADING
# =====================================================================

def load_fixtures_and_odds(start_date: str) -> pd.DataFrame:
    """
    Laad matches + odds via eci_fixture_links_v bridge.
    
    Flow: snapshots -> eci_fixture_links_v -> fixtures + odds_values
    """
    query = text("""
        SELECT DISTINCT ON (v.fixture_id)
            v.fixture_id as match_id,
            v.date_utc as date,
            v.home_team,
            v.away_team,
            v.competition,
            s.home_rating,
            s.away_rating,
            f.home_goals,
            f.away_goals,
            o.market_key,
            o.label,
            o.odd,
            o.last_update as odds_snapshot_date,
            o.bookmaker_id
        FROM public.eci_data_snapshots s
        INNER JOIN public.eci_fixture_links_v v 
            ON s.home_team = v.home_team 
            AND s.away_team = v.away_team 
            AND s.competition = v.competition
        LEFT JOIN public.fixtures f ON v.fixture_id = f.fixture_id
        LEFT JOIN public.odds_values o ON v.fixture_id = o.fixture_id
        WHERE v.date_utc::date >= :start_date
        ORDER BY v.fixture_id, o.last_update DESC NULLS LAST
    """)
    
    try:
        with db_engine().connect() as conn:
            df = pd.read_sql(query, conn, params={"start_date": start_date})
    except Exception as e:
        print(f"[ERROR] Could not load fixtures: {e}")
        return pd.DataFrame()
    
    if df.empty:
        print("[WARNING] No fixtures found in date range.")
        return df
    
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["odds_snapshot_date"] = pd.to_datetime(df["odds_snapshot_date"], utc=True).dt.tz_localize(None)
    
    # Convert ratings to numeric
    df["home_rating"] = pd.to_numeric(df["home_rating"], errors="coerce")
    df["away_rating"] = pd.to_numeric(df["away_rating"], errors="coerce")
    
    print(f"[fixtures] Geladen: {len(df)} fixture-odds records")
    print(f"[fixtures] Unique matches: {df['match_id'].nunique()}")
    print(f"[fixtures] Markets available: {df['market_key'].unique().tolist()}")
    
    return df


def match_movers_to_fixtures(movers: pd.DataFrame, fixtures: pd.DataFrame) -> list[MatchWithMover]:
    """
    For each mover, find matches they played in + odds available.
    
    Handle normalized odds: each odd is a row with market_key and label.
    """
    if movers.empty or fixtures.empty:
        print("[matching] Empty input (movers or fixtures)")
        return []
    
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    fixtures["odds_snapshot_date"] = pd.to_datetime(fixtures["odds_snapshot_date"])
    movers["date_delta"] = pd.to_datetime(movers["date_delta"])
    
    # Deduplicate fixtures (take latest odds per market per fixture)
    # Group by match_id and get latest odds_snapshot_date per market per bookmaker
    fixtures_dedup = fixtures.sort_values("odds_snapshot_date", na_position="last").drop_duplicates(
        subset=["match_id", "market_key", "label", "bookmaker_id"],
        keep="last"
    )
    
    results = []
    
    for _, mover in movers.iterrows():
        team_name = mover["team_name"]
        delta_date = mover["date_delta"]
        delta_value = mover["delta"]
        direction = mover["direction"]
        
        # Find matches of this team in next 14 days after delta
        future_window = fixtures_dedup[
            (fixtures_dedup["date"] > delta_date) &
            (fixtures_dedup["date"] <= delta_date + timedelta(days=14))
        ]
        
        # Find matches where this team plays (home or away)
        matches = future_window[
            (future_window["home_team"] == team_name) |
            (future_window["away_team"] == team_name)
        ]
        
        for match_id, match_group in matches.groupby("match_id"):
            # Get one snapshot of this match (all odds from same time)
            match_latest = match_group.iloc[0].copy()
            
            home_team = match_latest["home_team"]
            away_team = match_latest["away_team"]
            home_is_mover = home_team == team_name
            away_is_mover = away_team == team_name
            
            # Extract 1x2 odds from this match
            match_1x2 = match_group[match_group["market_key"] == "1x2"]
            
            if match_1x2.empty:
                # Try other markets or skip
                continue
            
            # Pivot odds by label to get home/draw/away
            odds_dict = dict(zip(match_1x2["label"], match_1x2["odd"]))
            home_odds = odds_dict.get("1", odds_dict.get("home"))
            draw_odds = odds_dict.get("X", odds_dict.get("draw"))
            away_odds = odds_dict.get("2", odds_dict.get("away"))
            
            odds_age_days = (match_latest["date"] - match_latest["odds_snapshot_date"]).days if pd.notna(match_latest["odds_snapshot_date"]) else None
            
            # Skip if odds missing or out of staleness window
            if odds_age_days is None or not (ODDS_STALENESS_MIN <= odds_age_days <= ODDS_STALENESS_MAX):
                continue
            
            # Skip if odds out of range
            if pd.notna(home_odds) and (home_odds < MIN_ODDS or home_odds > MAX_ODDS):
                continue
            if pd.notna(away_odds) and (away_odds < MIN_ODDS or away_odds > MAX_ODDS):
                continue
            
            # Determine result
            result = None
            if pd.notna(match_latest["home_goals"]) and pd.notna(match_latest["away_goals"]):
                if match_latest["home_goals"] > match_latest["away_goals"]:
                    result = "home_win"
                elif match_latest["away_goals"] > match_latest["home_goals"]:
                    result = "away_win"
                else:
                    result = "draw"
            
            # Select side for betting
            selected_side = None
            selected_odds = None
            
            if direction == "up":
                if home_is_mover and pd.notna(home_odds):
                    selected_side = "home"
                    selected_odds = home_odds
                elif away_is_mover and pd.notna(away_odds):
                    selected_side = "away"
                    selected_odds = away_odds
            
            if selected_side and pd.notna(selected_odds):
                won = None
                if result is not None:
                    if selected_side == "home":
                        won = result == "home_win"
                    else:
                        won = result == "away_win"
                
                results.append(
                    MatchWithMover(
                        match_id=match_id,
                        date=match_latest["date"],
                        home_team=home_team,
                        away_team=away_team,
                        competition=match_latest["competition"],
                        home_is_mover=home_is_mover,
                        home_delta=delta_value if home_is_mover else 0,
                        away_is_mover=away_is_mover,
                        away_delta=delta_value if away_is_mover else 0,
                        home_odds=home_odds,
                        away_odds=away_odds,
                        draw_odds=draw_odds,
                        odds_age_days=odds_age_days,
                        result=result,
                        selected_side=selected_side,
                        selected_odds=selected_odds,
                        won=won,
                    )
                )
    
    return results


# =====================================================================
# BACKTEST
# =====================================================================

def backtest_movers(matches_with_movers: list[MatchWithMover]) -> pd.DataFrame:
    """
    Analyze opportunities identified for rating movers.
    
    In early phase, we won't have match results yet, so focus on:
    - Which movers were identified
    - Which opportunities exist
    - What would happen if we bet on them (projection)
    """
    if not matches_with_movers:
        print("[backtest] Geen matches met movers gevonden.")
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            "match_id": m.match_id,
            "date": m.date,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "competition": m.competition,
            "side": m.selected_side,
            "odds": m.selected_odds,
            "odds_age": m.odds_age_days,
            "result": m.result,
            "won": m.won,
            "delta": m.home_delta if m.home_is_mover else m.away_delta,
        }
        for m in matches_with_movers
    ])
    
    print(f"\n[BACKTEST RESULTS]")
    print(f"  Total opportunities:  {len(df)}")
    
    # Check if we have results
    df_settled = df[df["won"].notna()].copy()
    
    if len(df_settled) == 0:
        print(f"  Settled matches:      0 (no results yet)")
        print(f"\n[status] Opportunities identified but matches not yet settled.")
        print(f"[next] Come back after matches play to see actual results.")
        return df
    
    # Calculate if we have results
    df_settled["stake"] = 1.0
    df_settled["won_binary"] = df_settled["won"].astype(float)
    df_settled["payout"] = df_settled["stake"] * df_settled["odds"] * df_settled["won_binary"]
    df_settled["profit"] = df_settled["payout"] - df_settled["stake"]
    
    wins = int(df_settled["won"].sum())
    losses = len(df_settled) - wins
    total_stake = df_settled["stake"].sum()
    total_payout = df_settled["payout"].sum()
    total_profit = df_settled["profit"].sum()
    roi_pct = (total_profit / total_stake * 100) if total_stake > 0 else 0
    hit_rate_pct = (wins / len(df_settled) * 100) if len(df_settled) > 0 else 0
    avg_odds = df_settled["odds"].mean()
    
    print(f"  Settled matches:      {len(df_settled)}")
    print(f"  Wins:                 {wins} ({hit_rate_pct:.1f}%)")
    print(f"  Losses:               {losses}")
    print(f"  Profit/Loss:          {total_profit:+.2f}")
    print(f"  ROI:                  {roi_pct:+.2f}%")
    print(f"  Avg odds:             {avg_odds:.2f}")
    
    breakeven_rate = (1.0 / avg_odds) * 100 if avg_odds > 0 else 0
    print(f"  Break-even:           {breakeven_rate:.1f}%")
    print(f"  vs actual:            {hit_rate_pct:.1f}% {'✓' if hit_rate_pct >= breakeven_rate else '✗'}")
    
    return df


# =====================================================================
# MAIN
# =====================================================================

def main(export_csv: bool = False, verbose: bool = False):
    print("\n" + "="*70)
    print("RATING MOMENTUM ENGINE - Market Lag Arbitrage Research")
    print("="*70)
    print(f"Config:")
    print(f"  Rating delta threshold:  ±{RATING_DELTA_THRESHOLD} points")
    print(f"  Lookback window:         {RATING_LOOKBACK_DAYS} days (auto-adjusted)")
    print(f"  Odds staleness window:   {ODDS_STALENESS_MIN}–{ODDS_STALENESS_MAX} days")
    print(f"  Odds range:              {MIN_ODDS}–{MAX_ODDS}")
    print(f"  Future window:           14 days after delta")
    print("="*70 + "\n")
    
    # Step 1: Load snapshots
    snapshots = load_eci_snapshots()
    if snapshots.empty:
        print("\n[FATAL] Snapshots are empty. Cannot proceed.")
        return pd.DataFrame()
    
    # Step 2: Calculate deltas
    deltas = calculate_rating_deltas(snapshots, lookback_days=RATING_LOOKBACK_DAYS)
    print(f"[deltas] Calculated {len(deltas)} rating changes")
    
    # Step 3: Identify movers
    movers = identify_movers(deltas, threshold=RATING_DELTA_THRESHOLD)
    
    if movers.empty:
        print(f"[movers] Geen movers gevonden met threshold ±{RATING_DELTA_THRESHOLD}")
        print("[info] Probeer: RATING_DELTA_THRESHOLD lager zetten (bijv. 100 i.p.v. 150)")
        return pd.DataFrame()
    
    print(f"[summary] Movers per direction:")
    print(f"  Up:   {(movers['direction'] == 'up').sum()}")
    print(f"  Down: {(movers['direction'] == 'down').sum()}")
    
    # Step 4: Load fixtures & odds
    fixtures = load_fixtures_and_odds(BACKTEST_START_DATE)
    if fixtures.empty:
        print("[FATAL] No fixtures found.")
        return pd.DataFrame()
    
    # Step 5: Match movers to fixtures
    print("\n[matching] Matching movers to fixtures...")
    matches_with_movers = match_movers_to_fixtures(movers, fixtures)
    print(f"[matching] Found {len(matches_with_movers)} mover-related opportunities")
    
    if not matches_with_movers:
        print("[WARNING] No matches between movers and fixtures. Data too recent?")
        return pd.DataFrame()
    
    # Step 6: Backtest
    print("\n" + "="*70)
    backtest_df = backtest_movers(matches_with_movers)
    
    if not backtest_df.empty:
        print("\n[sample] First 10 opportunities:")
        sample_cols = [c for c in ["date", "home_team", "away_team", "side", "odds", "delta"] if c in backtest_df.columns]
        print(backtest_df[sample_cols].head(10).to_string(index=False))
    
    # Step 7: Export
    if export_csv and not backtest_df.empty:
        csv_path = EXPORT_DIR / f"momentum_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        backtest_df.to_csv(csv_path, index=False)
        print(f"\n[export] CSV saved: {csv_path}")
    
    print("\n" + "="*70)
    print("Research complete.")
    print("="*70 + "\n")
    
    return backtest_df


def diagnose_schema():
    """Controleer welke tabellen en kolommen beschikbaar zijn."""
    print("\n" + "="*70)
    print("SCHEMA DIAGNOSTICS")
    print("="*70 + "\n")
    
    tables_to_check = [
        ("public", "eci_data_snapshots"),
        ("public", "eci_fixture_links_v"),
        ("public", "odds_values"),
    ]
    
    for schema, table_name in tables_to_check:
        query = text(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ordinal_position
        """)
        
        try:
            with db_engine().connect() as conn:
                result = conn.execute(query, {"schema": schema, "table": table_name})
                columns = result.fetchall()
            
            if columns:
                print(f"✓ {schema}.{table_name}")
                for col_name, data_type in columns:
                    print(f"    {col_name}: {data_type}")
            else:
                print(f"✗ {schema}.{table_name} NOT FOUND")
        except Exception as e:
            print(f"✗ {schema}.{table_name}: ERROR - {e}")
        
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rating Momentum Research")
    parser.add_argument("--export-csv", action="store_true", help="Export backtest results to CSV")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--diagnose", action="store_true", help="Check database schema and exit")
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_schema()
    else:
        main(export_csv=args.export_csv, verbose=args.verbose)