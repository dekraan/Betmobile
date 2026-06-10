"""
research_backtest_v4.py

Researchlaag voor Betmobile.

Doel:
- Laat de dagelijkse productieflow met run_model.py / eci_picks.py ongemoeid.
- Analyseer eerst opgeslagen picks uit public.picks_evaluated.
- Houd daarnaast een historische backtestmodus beschikbaar voor een brede ECI+odds+score view.

Aanbevolen eerste gebruik:
    python research_backtest.py

Dat is gelijk aan:
    python research_backtest.py --mode picks

Met CSV-export:
    python research_backtest.py --mode picks --export-csv

Brede historische backtest:
    python research_backtest.py --mode historical --source betmobile_tuning_preko_mv --no-refresh
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import (
    OUTPUT_DIR,
    RULE_MIN_PROB,
    RULE_MIN_VALUE,
    RULE_MIN_RATING_GAP,
    RULE_MIN_ODDS,
    RULE_MAX_ODDS,
    RULE_MIN_SNAPSHOTS,
    RULE_MIN_DRIFT_ABS,
    MIN_STRENGTH,
)
from db import db_engine, relation_exists, refresh_source_views
from rules import apply_rules, apply_drift
from picks import build_picks
from utils import choose_relevant_side, make_strength_bucket


# =====================================================================
# CONFIG
# =====================================================================

DEFAULT_SOURCE = "betmobile_tuning_preko_mv"
DEFAULT_SCHEMA = "public"
EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    name: str
    bets: int
    wins: int
    losses: int
    profit: float
    roi: float
    hitrate: float
    avg_odds: float | None


# =====================================================================
# GENERIC HELPERS
# =====================================================================

def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_result(result: BacktestResult) -> None:
    print(
        f"{result.name:<32} "
        f"bets={result.bets:>5}  "
        f"profit={result.profit:>8.2f}  "
        f"roi={result.roi:>7.2%}  "
        f"hit={result.hitrate:>7.2%}  "
        f"avg_odds={(result.avg_odds if result.avg_odds is not None else 0):>5.2f}"
    )


def summarize_bets(name: str, bets: pd.DataFrame) -> BacktestResult:
    if bets is None or bets.empty:
        return BacktestResult(name, 0, 0, 0, 0.0, 0.0, 0.0, None)

    n = len(bets)
    wins = int(bets["won"].sum())
    losses = int(n - wins)
    profit = float(bets["profit"].sum())
    roi = profit / n if n else 0.0
    hitrate = wins / n if n else 0.0
    avg_odds = float(bets["selected_odds"].mean()) if "selected_odds" in bets else None

    return BacktestResult(name, n, wins, losses, profit, roi, hitrate, avg_odds)


def print_table(title: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    print_header(title)
    if df is None or df.empty:
        print("Geen data.")
        return

    view = df.head(max_rows).copy()
    for col in view.columns:
        if col in {
            "profit", "roi", "hitrate", "avg_odds", "avg_strength",
            "max_strength", "avg_value_edge", "avg_prob_edge",
            "avg_drift_score", "avg_snapshots", "max_drawdown",
            "worst_month_roi", "best_month_roi", "monthly_roi_std",
            "last_50_roi", "min_rolling_roi_50", "max_rolling_roi_50",
            "profit_per_100_bets",
        }:
            view[col] = pd.to_numeric(view[col], errors="coerce").round(4)
    print(view.to_string(index=False))


def existing_relation_or_fail(source: str) -> None:
    q = text("""
        SELECT c.relkind
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = :source
    """)

    with db_engine().connect() as conn:
        row = conn.execute(q, {"source": source}).fetchone()

    if not row:
        raise RuntimeError(
            f"Bron public.{source} bestaat niet. Controleer de naam of geef een andere bron mee."
        )

    kind_map = {
        "r": "table",
        "v": "view",
        "m": "materialized_view",
        "p": "partitioned_table",
    }

    kind = kind_map.get(row[0], row[0])
    print(f"[source] public.{source} gevonden ({kind}).")


def get_table_columns(source: str, schema: str = DEFAULT_SCHEMA) -> set[str]:
    """
    Lees beschikbare kolommen uit PostgreSQL catalogus.

    Werkt voor tabellen, views en materialized views.
    """
    q = text(
        """
        SELECT a.attname AS column_name
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema
          AND c.relname = :source
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """
    )
    with db_engine().connect() as conn:
        rows = conn.execute(q, {"schema": schema, "source": source}).fetchall()
    return {r[0] for r in rows}


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], default=np.nan) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = default
    return df


# =====================================================================
# MODE 1: PICKS_EVALUATED RESEARCH
# =====================================================================

def load_picks_evaluated() -> pd.DataFrame:
    """Laad opgeslagen en gesettelde picks uit public.picks_evaluated."""
    existing_relation_or_fail("picks_evaluated")

    q = """
        SELECT
            run_id,
            match_id,
            competition,
            date,
            home_team,
            away_team,
            odds_home,
            odds_draw,
            odds_away,
            prob_home,
            prob_draw,
            prob_away,
            bet_home,
            bet_draw,
            bet_away,
            edge_h,
            edge_d,
            edge_a,
            rule_a,
            rule_b,
            rule_c,
            pick_reason,
            rule_strength,
            rule_strength_adj,
            away_drift_pct,
            home_drift_pct,
            n_snapshots,
            hours_stale,
            selection,
            score,
            result,
            outcome,
            settled_at,
            date_ts,
            rule_passed,
            rule_reason,
            pick_type,
            value_edge,
            prob_edge,
            drift_score,
            snapshot_count,
            strength_bucket,
            rating_gap,
            drift_range
        FROM public.picks_evaluated
        WHERE selection IS NOT NULL
          AND outcome IN ('WIN', 'LOSS')
    """

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    print(f"[picks_evaluated] settled rows loaded: {len(df)}")
    return df


def add_common_research_features(df: pd.DataFrame, kind: str = "picks") -> pd.DataFrame:
    """
    Voeg research-features/buckets toe die we in meerdere analyses kunnen gebruiken.

    kind='picks': verwacht columns zoals selection/prob_home/prob_away/selected_odds/rating_gap.
    kind='singlefail': verwacht columns zoals side/probability/odds/rating_gap/drift_pct.
    """
    df = df.copy()

    # Datumfeatures
    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date_dt"].dt.to_period("M").astype(str)
        df["weekday"] = df["date_dt"].dt.day_name()
        df["month_num"] = df["date_dt"].dt.month

        # Voetbalseizoenfase, grof en praktisch.
        # Aug-Sep = early, Oct-Feb = mid, Mar-May = late, Jun-Jul = summer/transition.
        df["season_phase"] = np.select(
            [
                df["month_num"].isin([8, 9]),
                df["month_num"].isin([10, 11, 12, 1, 2]),
                df["month_num"].isin([3, 4, 5]),
                df["month_num"].isin([6, 7]),
            ],
            ["early", "mid", "late", "summer"],
            default="unknown",
        )
    else:
        df["date_dt"] = pd.NaT
        df["month"] = "unknown"
        df["weekday"] = "unknown"
        df["season_phase"] = "unknown"

    # Geselecteerde probability
    if kind == "picks":
        for col in ["prob_home", "prob_draw", "prob_away"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["prob_selected"] = np.where(
            df["selection"] == "HOME",
            df.get("prob_home"),
            np.where(df["selection"] == "AWAY", df.get("prob_away"), df.get("prob_draw")),
        )
    else:
        if "probability" in df.columns:
            df["prob_selected"] = pd.to_numeric(df["probability"], errors="coerce")
        else:
            df["prob_selected"] = np.nan

    # Odds buckets
    df["odds_bucket"] = pd.cut(
        pd.to_numeric(df["selected_odds"], errors="coerce"),
        bins=[1.0, 1.4, 1.6, 1.8, 2.2, 3.0, 10.0],
        labels=["1.0-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.2", "2.2-3.0", "3.0+"],
    )

    # Probability buckets
    df["prob_bucket"] = pd.cut(
        pd.to_numeric(df["prob_selected"], errors="coerce"),
        bins=[0.0, 0.52, 0.55, 0.60, 0.65, 0.70, 1.0],
        labels=["<52%", "52-55%", "55-60%", "60-65%", "65-70%", "70%+"],
    )

    # Rating gap buckets
    if "rating_gap" in df.columns:
        df["rating_gap"] = pd.to_numeric(df["rating_gap"], errors="coerce")
    else:
        df["rating_gap"] = np.nan

    df["rating_gap_bucket"] = pd.cut(
        df["rating_gap"],
        bins=[0, 100, 250, 500, 1000, 5000, np.inf],
        labels=["0-100", "100-250", "250-500", "500-1000", "1000-5000", "5000+"],
        include_lowest=True,
    )

    # Snapshot buckets
    snap_col = "snapshot_count" if "snapshot_count" in df.columns else "n_snapshots"
    if snap_col in df.columns:
        df[snap_col] = pd.to_numeric(df[snap_col], errors="coerce")
        df["snapshots_used"] = df[snap_col]
    else:
        df["snapshots_used"] = np.nan

    df["snapshot_bucket"] = pd.cut(
        df["snapshots_used"],
        bins=[-0.1, 3, 6, 10, 15, 25, 50, np.inf],
        labels=["0-3", "4-6", "7-10", "11-15", "16-25", "26-50", "50+"],
    )

    # Strength buckets, met fallback per soort analyse.
    if kind == "picks":
        strength_col = "rule_strength_adj"
    else:
        strength_col = "single_fail_raw_strength"

    if strength_col in df.columns:
        df["research_strength"] = pd.to_numeric(df[strength_col], errors="coerce")
    else:
        df["research_strength"] = np.nan

    df["research_strength_bucket"] = df["research_strength"].apply(make_strength_bucket)

    # Market support: negatieve drift = odds dalen = support voor selectie.
    drift_source = "selected_drift_pct" if "selected_drift_pct" in df.columns else "drift_pct"
    if drift_source in df.columns:
        df["selected_drift_pct"] = pd.to_numeric(df[drift_source], errors="coerce")
    else:
        df["selected_drift_pct"] = np.nan

    df["market_support"] = np.select(
        [
            df["selected_drift_pct"] <= -0.03,
            df["selected_drift_pct"] >= 0.03,
        ],
        ["SUPPORT", "AGAINST"],
        default="NEUTRAL",
    )

    # Value score: voor picks meestal prob*odds of bet_home/bet_away; voor singlefails value_score.
    if "value_score" in df.columns:
        df["selected_value_score"] = pd.to_numeric(df["value_score"], errors="coerce")
    else:
        df["selected_value_score"] = pd.to_numeric(df["prob_selected"], errors="coerce") * pd.to_numeric(df["selected_odds"], errors="coerce")

    df["value_bucket"] = pd.cut(
        df["selected_value_score"],
        bins=[0, 0.95, 1.00, 1.04, 1.08, 1.15, 1.30, np.inf],
        labels=["<0.95", "0.95-1.00", "1.00-1.04", "1.04-1.08", "1.08-1.15", "1.15-1.30", "1.30+"],
    )

    return df


def build_calibration_table(df: pd.DataFrame, group_cols: list[str] | None = None, min_bets: int = 1) -> pd.DataFrame:
    """
    Calibration: verwachte probability vs echte hitrate.
    Dit zegt niet alleen of picks winstgevend zijn, maar of de probability zelf klopt.
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = group_cols or ["prob_bucket"]
    out = (
        df.groupby(group_cols, observed=True)
        .agg(
            bets=("profit", "size"),
            wins=("won", "sum"),
            profit=("profit", "sum"),
            avg_prob=("prob_selected", "mean"),
            avg_odds=("selected_odds", "mean"),
            avg_value=("selected_value_score", "mean"),
        )
        .reset_index()
    )
    out["hitrate"] = out["wins"] / out["bets"]
    out["roi"] = out["profit"] / out["bets"]
    out["calibration_error"] = out["hitrate"] - out["avg_prob"]
    out = out[out["bets"] >= min_bets].copy()
    return out.sort_values(["calibration_error", "roi"], ascending=[False, False])


def build_rolling_results(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Rolling resultaat per bet-volgorde op datum."""
    if df.empty:
        return pd.DataFrame()

    work = df.sort_values("date_dt").reset_index(drop=True).copy()
    work["bet_no"] = np.arange(1, len(work) + 1)
    work["cum_profit"] = work["profit"].cumsum()
    work["cum_roi"] = work["cum_profit"] / work["bet_no"]
    work[f"rolling_profit_{window}"] = work["profit"].rolling(window, min_periods=max(10, window // 3)).sum()
    work[f"rolling_roi_{window}"] = work["profit"].rolling(window, min_periods=max(10, window // 3)).mean()
    work[f"rolling_hitrate_{window}"] = work["won"].rolling(window, min_periods=max(10, window // 3)).mean()

    cols = [
        "bet_no", "date", "competition", "home_team", "away_team", "selection",
        "selected_odds", "prob_selected", "profit", "cum_profit", "cum_roi",
        f"rolling_profit_{window}", f"rolling_roi_{window}", f"rolling_hitrate_{window}",
    ]
    return work[[c for c in cols if c in work.columns]].copy()


def detect_danger_zones(tables: dict[str, pd.DataFrame], min_bets: int = 20, roi_threshold: float = -0.15) -> pd.DataFrame:
    """Verzamel segmenten die structureel zwak lijken."""
    rows = []
    for table_name, table in tables.items():
        if table is None or table.empty or "roi" not in table.columns or "bets" not in table.columns:
            continue
        bad = table[(table["bets"] >= min_bets) & (table["roi"] <= roi_threshold)].copy()
        if bad.empty:
            continue
        for _, r in bad.iterrows():
            dims = {c: r[c] for c in bad.columns if c not in {"bets", "wins", "profit", "roi", "hitrate", "avg_odds", "avg_strength", "avg_prob_selected", "avg_value_edge", "avg_prob_edge", "avg_drift_score", "avg_snapshots", "avg_prob", "avg_value", "calibration_error"}}
            rows.append({
                "table": table_name,
                "segment": " | ".join(f"{k}={v}" for k, v in dims.items()),
                "bets": int(r["bets"]),
                "profit": float(r.get("profit", 0)),
                "roi": float(r["roi"]),
                "hitrate": float(r.get("hitrate", np.nan)),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["roi", "bets"], ascending=[True, False])



def compute_stability_metrics(bets: pd.DataFrame, rolling_window: int = 50) -> dict[str, float]:
    """
    Extra stabiliteitsmetrics voor experimenten.

    - max_drawdown: grootste terugval vanaf een tussentijdse profit-piek, in units.
    - worst_month_roi: slechtste maand-ROI binnen het experiment.
    - positive_months / months: hoeveel maanden positief waren.
    - rolling ROI: hoe gedraagt het experiment zich over rollende blokken van N bets.
    """
    if bets is None or bets.empty:
        return {
            "max_drawdown": 0.0,
            "worst_month_roi": 0.0,
            "best_month_roi": 0.0,
            "monthly_roi_std": 0.0,
            "positive_months": 0,
            "months": 0,
            "last_50_roi": 0.0,
            "min_rolling_roi_50": 0.0,
            "max_rolling_roi_50": 0.0,
            "profit_per_100_bets": 0.0,
        }

    work = bets.copy()
    if "date_dt" in work.columns:
        work = work.sort_values("date_dt")
    elif "date" in work.columns:
        work["date_dt"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.sort_values("date_dt")

    work = work.reset_index(drop=True)
    work["cum_profit"] = work["profit"].cumsum()
    work["running_peak"] = work["cum_profit"].cummax().clip(lower=0)
    work["drawdown"] = work["cum_profit"] - work["running_peak"]
    max_drawdown = float(work["drawdown"].min()) if not work.empty else 0.0

    if "month" not in work.columns:
        work["month"] = pd.to_datetime(work.get("date", pd.NaT), errors="coerce").dt.to_period("M").astype(str)

    month_stats = (
        work.groupby("month", observed=True)
        .agg(bets=("profit", "size"), profit=("profit", "sum"))
        .reset_index()
    )
    month_stats = month_stats[month_stats["bets"] > 0].copy()
    if month_stats.empty:
        worst_month_roi = best_month_roi = monthly_roi_std = 0.0
        positive_months = months = 0
    else:
        month_stats["roi"] = month_stats["profit"] / month_stats["bets"]
        worst_month_roi = float(month_stats["roi"].min())
        best_month_roi = float(month_stats["roi"].max())
        monthly_roi_std = float(month_stats["roi"].std(ddof=0)) if len(month_stats) > 1 else 0.0
        positive_months = int((month_stats["roi"] > 0).sum())
        months = int(len(month_stats))

    roll = work["profit"].rolling(
        rolling_window,
        min_periods=max(10, rolling_window // 3),
    ).mean()
    if roll.dropna().empty:
        last_50_roi = min_rolling_roi_50 = max_rolling_roi_50 = 0.0
    else:
        last_50_roi = float(roll.dropna().iloc[-1])
        min_rolling_roi_50 = float(roll.min())
        max_rolling_roi_50 = float(roll.max())

    profit_per_100_bets = float(work["profit"].mean() * 100) if len(work) else 0.0

    return {
        "max_drawdown": max_drawdown,
        "worst_month_roi": worst_month_roi,
        "best_month_roi": best_month_roi,
        "monthly_roi_std": monthly_roi_std,
        "positive_months": positive_months,
        "months": months,
        "last_50_roi": last_50_roi,
        "min_rolling_roi_50": min_rolling_roi_50,
        "max_rolling_roi_50": max_rolling_roi_50,
        "profit_per_100_bets": profit_per_100_bets,
    }


def get_filter_experiments(base: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """
    Centrale definitie van filter-experimenten.

    Elk experiment geeft een keep-mask terug: True = pick blijft in het experiment.
    Zo kun je veilig research doen zonder productie aan te passen.
    """
    true_mask = pd.Series(True, index=base.index)
    odds = pd.to_numeric(base["selected_odds"], errors="coerce")
    strength = pd.to_numeric(base["rule_strength_adj"], errors="coerce")
    rating_gap = pd.to_numeric(base["rating_gap"], errors="coerce")
    prob_bucket = base["prob_bucket"].astype(str)
    odds_bucket = base["odds_bucket"].astype(str)
    market_support = base["market_support"].astype(str)
    pick_type = base["pick_type"].astype(str)

    no_main_strength_lt_2 = ~((pick_type == "MAIN") & (strength < 2.0))
    no_odds_22_30 = ~odds.between(2.2, 3.0, inclusive="left")
    no_prob_55_60 = prob_bucket != "55-60%"
    no_gap_lt_250 = rating_gap >= 250
    no_low_odds_high_prob = ~((odds_bucket == "1.4-1.6") & (prob_bucket == "70%+"))
    no_support_odds_22_30 = ~((market_support == "SUPPORT") & (odds_bucket == "2.2-3.0"))

    danger_v1 = no_main_strength_lt_2 & no_odds_22_30 & no_prob_55_60 & no_gap_lt_250
    danger_v2 = danger_v1 & no_low_odds_high_prob

    # Varianten die niet per se productieregels zijn, maar de sweetspots helpen isoleren.
    core_mid_odds = odds_bucket.isin(["1.6-1.8", "1.8-2.2"])
    core_prob = prob_bucket.isin(["60-65%", "65-70%", "70%+"])
    avoid_bad_calibration = ~(
        (prob_bucket == "55-60%")
        | ((prob_bucket == "70%+") & (odds_bucket == "1.4-1.6"))
    )

    return [
        ("baseline", "Geen extra filters; huidige opgeslagen picks.", true_mask),
        ("exclude_main_strength_lt_2", "Sluit MAIN picks met strength < 2 uit; SECONDARY blijft staan.", no_main_strength_lt_2),
        ("exclude_odds_2_2_to_3_0", "Sluit odds tussen 2.2 en 3.0 uit.", no_odds_22_30),
        ("exclude_prob_55_60", "Sluit probability bucket 55-60% uit.", no_prob_55_60),
        ("exclude_rating_gap_lt_250", "Sluit picks met rating_gap < 250 uit.", no_gap_lt_250),
        ("exclude_low_odds_high_prob", "Sluit odds 1.4-1.6 + probability 70%+ uit.", no_low_odds_high_prob),
        ("exclude_support_odds_2_2_to_3_0", "Sluit market SUPPORT + odds 2.2-3.0 uit.", no_support_odds_22_30),
        ("avoid_bad_calibration_v1", "Sluit 55-60% uit en sluit 1.4-1.6 + 70%+ uit.", avoid_bad_calibration),
        ("core_mid_odds_only", "Alleen odds 1.6-2.2; onderzoekt de huidige sweetspot.", core_mid_odds),
        ("core_prob_60_plus", "Alleen probability >=60%; onderzoekt hogere ECI-confidence zonder 52-55 longshot-effect.", core_prob),
        ("danger_combo_v1", "Combineert: geen MAIN strength <2, geen odds 2.2-3.0, geen prob 55-60%, geen rating_gap <250.", danger_v1),
        ("danger_combo_v2", "V1 + sluit odds 1.4-1.6 met probability 70%+ uit.", danger_v2),
        ("danger_combo_v2_no_longshots", "V2 + sluit odds 3.0+ uit om longshot-variantie niet mee te tellen.", danger_v2 & (odds < 3.0)),
    ]


def summarize_filter_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simuleer filter-experimenten op de al opgeslagen picks.

    Belangrijk:
    - Dit verandert niets aan productie.
    - Het beantwoordt alleen: wat was er historisch gebeurd als we segment X hadden uitgesloten?
    - Gebruik dit voorlopig als richtinggevend onderzoek, niet als harde waarheid.
    """
    if df.empty:
        return pd.DataFrame()

    base = df.copy()
    base_bets = len(base)
    base_profit = float(base["profit"].sum())
    base_roi = base_profit / base_bets if base_bets else 0.0

    def clean_mask(mask) -> pd.Series:
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=base.index)
        return mask.reindex(base.index).fillna(False).astype(bool)

    rows = []
    for name, description, mask in get_filter_experiments(base):
        keep = clean_mask(mask)
        part = base[keep].copy()

        bets = len(part)
        wins = int(part["won"].sum()) if bets else 0
        profit = float(part["profit"].sum()) if bets else 0.0
        roi = profit / bets if bets else 0.0
        hitrate = wins / bets if bets else 0.0
        avg_odds = float(part["selected_odds"].mean()) if bets else np.nan
        stability = compute_stability_metrics(part, rolling_window=50)

        rows.append(
            {
                "experiment": name,
                "bets": bets,
                "removed_bets": base_bets - bets,
                "kept_pct": bets / base_bets if base_bets else 0.0,
                "wins": wins,
                "profit": profit,
                "profit_delta": profit - base_profit,
                "roi": roi,
                "roi_delta": roi - base_roi,
                "hitrate": hitrate,
                "avg_odds": avg_odds,
                **stability,
                "description": description,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["roi", "profit"], ascending=[False, False])


def build_experiment_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detailbestand: per pick per experiment of hij behouden zou blijven.
    Handig voor export naar Excel/PowerBI en inspectie van weggefilterde wedstrijden.
    """
    if df.empty:
        return pd.DataFrame()

    base_cols = [
        "match_id", "date", "competition", "home_team", "away_team", "selection",
        "pick_type", "selected_odds", "prob_selected", "selected_value_score",
        "rating_gap", "strength_bucket", "prob_bucket", "odds_bucket", "market_support",
        "outcome", "profit",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    rows = []
    for name, description, mask in get_filter_experiments(df):
        keep = mask.reindex(df.index).fillna(False).astype(bool)
        part = df[base_cols].copy()
        part["experiment"] = name
        part["experiment_description"] = description
        part["kept"] = keep.values
        rows.append(part)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_picks_evaluated(df: pd.DataFrame) -> pd.DataFrame:
    """Maak opgeslagen picks geschikt voor analyse."""
    df = df.copy()

    numeric_cols = [
        "odds_home", "odds_draw", "odds_away",
        "prob_home", "prob_draw", "prob_away",
        "rule_strength", "rule_strength_adj",
        "away_drift_pct", "home_drift_pct",
        "n_snapshots", "hours_stale",
        "value_edge", "prob_edge", "drift_score",
        "snapshot_count", "rating_gap", "drift_range",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["selected_odds"] = np.select(
        [
            df["selection"] == "HOME",
            df["selection"] == "AWAY",
            df["selection"] == "DRAW",
        ],
        [df["odds_home"], df["odds_away"], df["odds_draw"]],
        default=np.nan,
    )

    df["won"] = df["outcome"] == "WIN"
    df["profit"] = np.where(df["won"], df["selected_odds"] - 1.0, -1.0)

    if "pick_type" not in df.columns:
        df["pick_type"] = "UNKNOWN"
    df["pick_type"] = df["pick_type"].fillna("UNKNOWN")

    if "strength_bucket" not in df.columns:
        df["strength_bucket"] = df["rule_strength_adj"].apply(make_strength_bucket)
    else:
        fallback_bucket = df["rule_strength_adj"].apply(make_strength_bucket)
        df["strength_bucket"] = df["strength_bucket"].fillna(fallback_bucket)

    df["selected_drift_pct"] = np.where(
        df["selection"] == "HOME",
        df["home_drift_pct"],
        np.where(df["selection"] == "AWAY", df["away_drift_pct"], np.nan),
    )

    df["drift_bucket"] = pd.cut(
        df["selected_drift_pct"],
        bins=[-np.inf, -0.10, -0.05, -0.03, 0.00, 0.03, 0.05, 0.10, np.inf],
        labels=["<=-10%", "-10/-5%", "-5/-3%", "-3/0%", "0/3%", "3/5%", "5/10%", ">10%"],
    )

    df = add_common_research_features(df, kind="picks")
    return df


def summarize_picks_grouped(bets: pd.DataFrame, group_cols: list[str], min_bets: int = 1) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    out = (
        bets.groupby(group_cols, observed=True)
        .agg(
            bets=("profit", "size"),
            wins=("won", "sum"),
            profit=("profit", "sum"),
            avg_odds=("selected_odds", "mean"),
            avg_strength=("rule_strength_adj", "mean"),
            avg_prob_selected=("prob_selected", "mean"),
            avg_value=("selected_value_score", "mean"),
            avg_value_edge=("value_edge", "mean"),
            avg_prob_edge=("prob_edge", "mean"),
            avg_drift_score=("drift_score", "mean"),
            avg_drift_pct=("selected_drift_pct", "mean"),
            avg_snapshots=("snapshots_used", "mean"),
        )
        .reset_index()
    )
    out["roi"] = out["profit"] / out["bets"]
    out["hitrate"] = out["wins"] / out["bets"]
    out = out[out["bets"] >= min_bets].copy()
    return out.sort_values(["roi", "bets"], ascending=[False, False])


def run_picks_evaluated_research(export_csv: bool = False) -> pd.DataFrame:
    print_header("BETMOBILE PICKS_EVALUATED RESEARCH")

    raw = load_picks_evaluated()
    df = prepare_picks_evaluated(raw)

    if df.empty:
        print("Geen settled picks gevonden in public.picks_evaluated.")
        return df

    print_header("OVERALL STORED PICK RESULTS")
    for pick_type, part in df.groupby("pick_type", observed=True):
        print_result(summarize_bets(str(pick_type), part))
    print_result(summarize_bets("ALL STORED PICKS", df))

    # Basis segmenten
    by_type_strength = summarize_picks_grouped(df, ["pick_type", "strength_bucket"])
    by_comp = summarize_picks_grouped(df, ["competition"], min_bets=10)
    by_comp_type = summarize_picks_grouped(df, ["competition", "pick_type"], min_bets=10)
    by_drift = summarize_picks_grouped(df, ["drift_bucket"], min_bets=10)
    by_selection = summarize_picks_grouped(df, ["selection"])
    by_odds = summarize_picks_grouped(df, ["odds_bucket"], min_bets=10)
    by_prob = summarize_picks_grouped(df, ["prob_bucket"], min_bets=10)
    by_rating_gap = summarize_picks_grouped(df, ["rating_gap_bucket"], min_bets=10)
    by_snapshots = summarize_picks_grouped(df, ["snapshot_bucket"], min_bets=10)
    by_market_support = summarize_picks_grouped(df, ["market_support"], min_bets=10)
    by_value_bucket = summarize_picks_grouped(df, ["value_bucket"], min_bets=10)

    # Kruistabellen: hier zit vaak de echte edge.
    by_odds_prob = summarize_picks_grouped(df, ["odds_bucket", "prob_bucket"], min_bets=10)
    by_odds_rating = summarize_picks_grouped(df, ["odds_bucket", "rating_gap_bucket"], min_bets=10)
    by_prob_rating = summarize_picks_grouped(df, ["prob_bucket", "rating_gap_bucket"], min_bets=10)
    by_support_prob = summarize_picks_grouped(df, ["market_support", "prob_bucket"], min_bets=10)
    by_support_odds = summarize_picks_grouped(df, ["market_support", "odds_bucket"], min_bets=10)

    # Tijdanalyse
    by_month = summarize_picks_grouped(df, ["month"], min_bets=10)
    by_weekday = summarize_picks_grouped(df, ["weekday"], min_bets=10)
    by_season_phase = summarize_picks_grouped(df, ["season_phase"], min_bets=10)

    # Calibration
    calibration = build_calibration_table(df, ["prob_bucket"], min_bets=10)
    calibration_by_type = build_calibration_table(df, ["pick_type", "prob_bucket"], min_bets=10)

    # Special: longshots apart bekijken.
    longshots = df[df["selected_odds"] >= 3.0].copy()
    by_longshot_comp = summarize_picks_grouped(longshots, ["competition"], min_bets=1)
    by_longshot_prob = summarize_picks_grouped(longshots, ["prob_bucket"], min_bets=1)

    # Rolling resultaat
    rolling = build_rolling_results(df, window=50)

    tables = {
        "stored_picks_by_type_strength": by_type_strength,
        "stored_picks_by_comp": by_comp,
        "stored_picks_by_comp_type": by_comp_type,
        "stored_picks_by_drift": by_drift,
        "stored_picks_by_selection": by_selection,
        "stored_picks_by_odds": by_odds,
        "stored_picks_by_prob": by_prob,
        "stored_picks_by_rating_gap": by_rating_gap,
        "stored_picks_by_snapshots": by_snapshots,
        "stored_picks_by_market_support": by_market_support,
        "stored_picks_by_value_bucket": by_value_bucket,
        "stored_picks_cross_odds_prob": by_odds_prob,
        "stored_picks_cross_odds_rating": by_odds_rating,
        "stored_picks_cross_prob_rating": by_prob_rating,
        "stored_picks_cross_support_prob": by_support_prob,
        "stored_picks_cross_support_odds": by_support_odds,
        "stored_picks_by_month": by_month,
        "stored_picks_by_weekday": by_weekday,
        "stored_picks_by_season_phase": by_season_phase,
        "stored_picks_calibration": calibration,
        "stored_picks_calibration_by_type": calibration_by_type,
        "stored_picks_longshots_by_comp": by_longshot_comp,
        "stored_picks_longshots_by_prob": by_longshot_prob,
    }

    danger_zones = detect_danger_zones(tables, min_bets=20, roi_threshold=-0.15)
    tables["stored_picks_danger_zones"] = danger_zones

    baseline_roi = summarize_bets("baseline", df).roi
    best_segments, worst_segments, interesting_segments = discover_segments(
        df,
        baseline_roi=baseline_roi,
        min_bets=20,
    )
    tables["stored_picks_auto_best_segments"] = best_segments
    tables["stored_picks_auto_worst_segments"] = worst_segments
    tables["stored_picks_auto_interesting_segments"] = interesting_segments

    filter_experiments = summarize_filter_experiments(df)
    experiment_detail = build_experiment_detail(df)
    tables["stored_picks_filter_experiments"] = filter_experiments

    print_table("PICKS BY TYPE + STRENGTH", by_type_strength)
    print_table("PICKS BY COMPETITION (min 10)", by_comp)
    print_table("PICKS BY COMPETITION + TYPE (min 10)", by_comp_type)
    print_table("PICKS BY DRIFT BUCKET (min 10)", by_drift)
    print_table("PICKS BY SELECTION", by_selection)
    print_table("PICKS BY ODDS BUCKET", by_odds)
    print_table("PICKS BY PROBABILITY BUCKET", by_prob)
    print_table("PICKS BY RATING GAP BUCKET", by_rating_gap)
    print_table("PICKS BY SNAPSHOT BUCKET", by_snapshots)
    print_table("PICKS BY MARKET SUPPORT", by_market_support)
    print_table("PICKS BY VALUE BUCKET", by_value_bucket)

    print_table("CROSS: ODDS BUCKET + PROBABILITY BUCKET", by_odds_prob)
    print_table("CROSS: ODDS BUCKET + RATING GAP", by_odds_rating)
    print_table("CROSS: PROBABILITY BUCKET + RATING GAP", by_prob_rating)
    print_table("CROSS: MARKET SUPPORT + PROBABILITY", by_support_prob)
    print_table("CROSS: MARKET SUPPORT + ODDS", by_support_odds)

    print_table("TIME: PICKS BY MONTH", by_month)
    print_table("TIME: PICKS BY WEEKDAY", by_weekday)
    print_table("TIME: PICKS BY SEASON PHASE", by_season_phase)

    print_table("CALIBRATION: PROBABILITY VS HITRATE", calibration)
    print_table("CALIBRATION: PICK TYPE + PROBABILITY", calibration_by_type)

    print_table("SPECIAL: LONGSHOTS BY COMPETITION", by_longshot_comp)
    print_table("SPECIAL: LONGSHOTS BY PROBABILITY", by_longshot_prob)
    print_table("DANGER ZONES", danger_zones)

    print_table("AUTO SEGMENT DISCOVERY: BEST SEGMENTS", best_segments, max_rows=40)
    print_table("AUTO SEGMENT DISCOVERY: WORST SEGMENTS", worst_segments, max_rows=40)
    print_table("AUTO SEGMENT DISCOVERY: INTERESTING VS BASELINE", interesting_segments, max_rows=80)
    print_table("FILTER EXPERIMENTS", filter_experiments, max_rows=20)

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        detail_path = EXPORT_DIR / f"stored_picks_detail_{stamp}.csv"
        df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"[export] {detail_path}")

        if rolling is not None and not rolling.empty:
            path = EXPORT_DIR / f"stored_picks_rolling_{stamp}.csv"
            rolling.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[export] {path}")

        if experiment_detail is not None and not experiment_detail.empty:
            path = EXPORT_DIR / f"stored_picks_experiment_detail_{stamp}.csv"
            experiment_detail.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[export] {path}")

        for name, table in tables.items():
            if table is not None and not table.empty:
                path = EXPORT_DIR / f"{name}_{stamp}.csv"
                table.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"[export] {path}")

    return df






# =====================================================================
# AUTO SEGMENT DISCOVERY
# =====================================================================

def discover_segments(
    df: pd.DataFrame,
    baseline_roi: float | None = None,
    min_bets: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Zoek automatisch naar opvallend sterke/zwakke segmenten.

    Dit is bewust breed, maar niet onbeperkt:
    - alleen segmenten met minimaal min_bets
    - combinaties van 1, 2 en enkele 3 dimensies
    - bedoeld voor research, niet direct voor productie-regels
    """
    if df is None or df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    if baseline_roi is None:
        baseline_result = summarize_bets("baseline", df)
        baseline_roi = baseline_result.roi

    candidate_dimensions = [
        "pick_type",
        "selection",
        "competition",
        "strength_bucket",
        "odds_bucket",
        "prob_bucket",
        "rating_gap_bucket",
        "snapshot_bucket",
        "market_support",
        "value_bucket",
        "drift_bucket",
        "season_phase",
        "weekday",
        "month",
    ]
    dims = [d for d in candidate_dimensions if d in df.columns]

    group_sets = []

    # 1D
    for d in dims:
        group_sets.append([d])

    # 2D kerncombinaties
    preferred_2d = [
        ["prob_bucket", "odds_bucket"],
        ["prob_bucket", "rating_gap_bucket"],
        ["prob_bucket", "value_bucket"],
        ["prob_bucket", "market_support"],
        ["prob_bucket", "strength_bucket"],
        ["odds_bucket", "rating_gap_bucket"],
        ["odds_bucket", "value_bucket"],
        ["odds_bucket", "market_support"],
        ["odds_bucket", "strength_bucket"],
        ["rating_gap_bucket", "value_bucket"],
        ["rating_gap_bucket", "market_support"],
        ["pick_type", "prob_bucket"],
        ["pick_type", "odds_bucket"],
        ["pick_type", "rating_gap_bucket"],
        ["pick_type", "strength_bucket"],
        ["competition", "prob_bucket"],
        ["competition", "odds_bucket"],
        ["competition", "pick_type"],
        ["market_support", "value_bucket"],
        ["snapshot_bucket", "prob_bucket"],
        ["snapshot_bucket", "odds_bucket"],
        ["weekday", "prob_bucket"],
    ]
    for gs in preferred_2d:
        if all(g in dims for g in gs):
            group_sets.append(gs)

    # 3D alleen voor belangrijkste modeldimensies, anders wordt het te noisy.
    preferred_3d = [
        ["prob_bucket", "odds_bucket", "rating_gap_bucket"],
        ["prob_bucket", "odds_bucket", "market_support"],
        ["prob_bucket", "odds_bucket", "pick_type"],
        ["prob_bucket", "rating_gap_bucket", "pick_type"],
        ["odds_bucket", "rating_gap_bucket", "pick_type"],
        ["prob_bucket", "odds_bucket", "value_bucket"],
        ["prob_bucket", "odds_bucket", "strength_bucket"],
    ]
    for gs in preferred_3d:
        if all(g in dims for g in gs):
            group_sets.append(gs)

    rows = []
    seen = set()

    for group_cols in group_sets:
        key = tuple(group_cols)
        if key in seen:
            continue
        seen.add(key)

        try:
            table = summarize_picks_grouped(df, group_cols, min_bets=min_bets)
        except Exception:
            continue

        if table is None or table.empty:
            continue

        for _, r in table.iterrows():
            segment_parts = [f"{c}={r.get(c)}" for c in group_cols]

            rows.append({
                "dimensions": " + ".join(group_cols),
                "segment": " | ".join(segment_parts),
                "bets": int(r.get("bets", 0)),
                "wins": int(r.get("wins", 0)),
                "profit": float(r.get("profit", 0.0)),
                "roi": float(r.get("roi", 0.0)),
                "roi_vs_baseline": float(r.get("roi", 0.0)) - float(baseline_roi),
                "hitrate": float(r.get("hitrate", 0.0)),
                "avg_odds": float(r.get("avg_odds", np.nan)) if pd.notna(r.get("avg_odds", np.nan)) else np.nan,
                "avg_prob_selected": float(r.get("avg_prob_selected", np.nan)) if pd.notna(r.get("avg_prob_selected", np.nan)) else np.nan,
                "avg_value": float(r.get("avg_value", np.nan)) if pd.notna(r.get("avg_value", np.nan)) else np.nan,
                "avg_strength": float(r.get("avg_strength", np.nan)) if pd.notna(r.get("avg_strength", np.nan)) else np.nan,
                "avg_snapshots": float(r.get("avg_snapshots", np.nan)) if pd.notna(r.get("avg_snapshots", np.nan)) else np.nan,
            })

    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty

    all_segments = pd.DataFrame(rows)

    best_segments = (
        all_segments
        .sort_values(["roi", "bets"], ascending=[False, False])
        .head(40)
        .copy()
    )

    worst_segments = (
        all_segments
        .sort_values(["roi", "bets"], ascending=[True, False])
        .head(40)
        .copy()
    )

    interesting_segments = all_segments[
        (
            (all_segments["roi_vs_baseline"] >= 0.10)
            | (all_segments["roi_vs_baseline"] <= -0.10)
        )
        & (all_segments["bets"] >= min_bets)
    ].copy()

    if not interesting_segments.empty:
        interesting_segments["abs_delta"] = interesting_segments["roi_vs_baseline"].abs()
        interesting_segments = (
            interesting_segments
            .sort_values(["abs_delta", "bets"], ascending=[False, False])
            .drop(columns=["abs_delta"])
            .head(80)
            .copy()
        )

    return best_segments, worst_segments, interesting_segments


# =====================================================================
# MODE 2: SINGLE_FAIL CANDIDATES RESEARCH
# =====================================================================

def load_single_fail_candidates() -> pd.DataFrame:
    """Laad gesettelde single-fail kandidaten uit public.picks_single_fail_candidates."""
    existing_relation_or_fail("picks_single_fail_candidates")
    available = get_table_columns("picks_single_fail_candidates")

    wanted = [
        "run_id", "match_id", "date", "competition", "home_team", "away_team",
        "side", "fail_reason", "single_fail_margin", "snap_needed",
        "single_fail_raw_strength", "single_fail_adj_strength", "single_fail_calibrated_strength",
        "odds", "probability", "value_score",
        "odds_home", "odds_draw", "odds_away",
        "prob_home", "prob_draw", "prob_away",
        "bet_home", "bet_draw", "bet_away",
        "rating_gap", "rating_home_edge", "n_snapshots", "drift_pct",
        "home_drift_pct", "away_drift_pct",
        "home_fail_reasons", "away_fail_reasons", "home_fail_count", "away_fail_count",
        "score", "result", "outcome", "settled_at",
    ]
    cols = [c for c in wanted if c in available]

    required = ["match_id", "side", "fail_reason", "odds", "outcome"]
    missing = [c for c in required if c not in available]
    if missing:
        raise RuntimeError(
            "Deze verplichte kolommen ontbreken in picks_single_fail_candidates: "
            + ", ".join(missing)
        )

    select_list = ",\n            ".join(cols)
    q = f"""
        SELECT
            {select_list}
        FROM public.picks_single_fail_candidates
        WHERE side IS NOT NULL
          AND outcome IN ('WIN', 'LOSS')
    """

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    print(f"[single_fail_candidates] settled rows loaded: {len(df)}")
    return df


def prepare_single_fail_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Maak single-fail kandidaten geschikt voor analyse."""
    df = df.copy()

    numeric_cols = [
        "single_fail_margin", "snap_needed", "single_fail_raw_strength",
        "single_fail_adj_strength", "single_fail_calibrated_strength",
        "odds", "probability", "value_score",
        "odds_home", "odds_draw", "odds_away",
        "prob_home", "prob_draw", "prob_away",
        "bet_home", "bet_draw", "bet_away",
        "rating_gap", "rating_home_edge", "n_snapshots", "drift_pct",
        "home_drift_pct", "away_drift_pct", "home_fail_count", "away_fail_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["selected_odds"] = df["odds"]
    df["selection"] = df["side"]
    df["won"] = df["outcome"] == "WIN"
    df["profit"] = np.where(df["won"], df["selected_odds"] - 1.0, -1.0)

    if "single_fail_raw_strength" not in df.columns:
        df["single_fail_raw_strength"] = np.nan

    df["strength_bucket"] = df["single_fail_raw_strength"].apply(make_strength_bucket)

    if "drift_pct" not in df.columns:
        df["drift_pct"] = np.nan

    df["drift_bucket"] = pd.cut(
        df["drift_pct"],
        bins=[-np.inf, -0.10, -0.05, -0.03, 0.00, 0.03, 0.05, 0.10, np.inf],
        labels=["<=-10%", "-10/-5%", "-5/-3%", "-3/0%", "0/3%", "3/5%", "5/10%", ">10%"],
    )

    df = add_common_research_features(df, kind="singlefail")
    return df


def summarize_single_fail_candidates_grouped(
    bets: pd.DataFrame,
    group_cols: list[str],
    min_bets: int = 1,
) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    agg = {
        "bets": ("profit", "size"),
        "wins": ("won", "sum"),
        "profit": ("profit", "sum"),
        "avg_odds": ("selected_odds", "mean"),
        "avg_strength": ("single_fail_raw_strength", "mean"),
        "max_strength": ("single_fail_raw_strength", "max"),
    }
    optional_aggs = {
        "avg_margin": ("single_fail_margin", "mean"),
        "avg_probability": ("probability", "mean"),
        "avg_value_score": ("value_score", "mean"),
        "avg_drift_pct": ("drift_pct", "mean"),
        "avg_snapshots": ("n_snapshots", "mean"),
    }
    for out_col, spec in optional_aggs.items():
        if spec[0] in bets.columns:
            agg[out_col] = spec

    out = bets.groupby(group_cols, observed=True).agg(**agg).reset_index()
    out["roi"] = out["profit"] / out["bets"]
    out["hitrate"] = out["wins"] / out["bets"]
    out = out[out["bets"] >= min_bets].copy()
    return out.sort_values(["roi", "bets"], ascending=[False, False])


def run_single_fail_research(export_csv: bool = False) -> pd.DataFrame:
    print_header("BETMOBILE SINGLE_FAIL CANDIDATES RESEARCH")

    raw = load_single_fail_candidates()
    df = prepare_single_fail_candidates(raw)

    if df.empty:
        print("Geen gesettelde single-fail kandidaten gevonden.")
        return df

    print_header("OVERALL SINGLE_FAIL RESULTS")
    print_result(summarize_bets("SINGLE_FAIL ALL", df))

    by_reason = summarize_single_fail_candidates_grouped(df, ["fail_reason"])
    by_reason_strength = summarize_single_fail_candidates_grouped(df, ["fail_reason", "strength_bucket"])
    by_comp_reason = summarize_single_fail_candidates_grouped(df, ["competition", "fail_reason"], min_bets=10)
    by_side = summarize_single_fail_candidates_grouped(df, ["side"])
    by_drift = summarize_single_fail_candidates_grouped(df, ["drift_bucket"], min_bets=10)
    by_odds = summarize_single_fail_candidates_grouped(df, ["odds_bucket"], min_bets=10)
    by_prob = summarize_single_fail_candidates_grouped(df, ["prob_bucket"], min_bets=10)
    by_rating_gap = summarize_single_fail_candidates_grouped(df, ["rating_gap_bucket"], min_bets=10)
    by_market_support = summarize_single_fail_candidates_grouped(df, ["market_support"], min_bets=10)
    by_odds_reason = summarize_single_fail_candidates_grouped(df, ["fail_reason", "odds_bucket"], min_bets=10)
    by_reason_support = summarize_single_fail_candidates_grouped(df, ["fail_reason", "market_support"], min_bets=10)

    print_table("SINGLE_FAIL BY REASON", by_reason)
    print_table("SINGLE_FAIL BY REASON + STRENGTH", by_reason_strength)
    print_table("SINGLE_FAIL BY COMPETITION + REASON (min 10)", by_comp_reason)
    print_table("SINGLE_FAIL BY SIDE", by_side)
    print_table("SINGLE_FAIL BY DRIFT BUCKET (min 10)", by_drift)
    print_table("SINGLE_FAIL BY ODDS BUCKET (min 10)", by_odds)
    print_table("SINGLE_FAIL BY PROBABILITY BUCKET (min 10)", by_prob)
    print_table("SINGLE_FAIL BY RATING GAP BUCKET (min 10)", by_rating_gap)
    print_table("SINGLE_FAIL BY MARKET SUPPORT (min 10)", by_market_support)
    print_table("SINGLE_FAIL CROSS: REASON + ODDS", by_odds_reason)
    print_table("SINGLE_FAIL CROSS: REASON + MARKET SUPPORT", by_reason_support)

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_path = EXPORT_DIR / f"single_fail_detail_{stamp}.csv"
        df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"[export] {detail_path}")

        tables = {
            "single_fail_by_reason": by_reason,
            "single_fail_by_reason_strength": by_reason_strength,
            "single_fail_by_comp_reason": by_comp_reason,
            "single_fail_by_side": by_side,
            "single_fail_by_drift": by_drift,
            "single_fail_by_odds": by_odds,
            "single_fail_by_prob": by_prob,
            "single_fail_by_rating_gap": by_rating_gap,
            "single_fail_by_market_support": by_market_support,
            "single_fail_cross_reason_odds": by_odds_reason,
            "single_fail_cross_reason_support": by_reason_support,
        }
        for name, table in tables.items():
            if table is not None and not table.empty:
                path = EXPORT_DIR / f"{name}_{stamp}.csv"
                table.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"[export] {path}")
    return df


def run_all_research(export_csv: bool = False) -> None:
    """Draai picks_evaluated en daarna, als beschikbaar, single-fail candidates."""
    run_picks_evaluated_research(export_csv=export_csv)

    try:
        existing_relation_or_fail("picks_single_fail_candidates")
    except RuntimeError:
        print_header("SINGLE_FAIL CANDIDATES")
        print("Tabel public.picks_single_fail_candidates bestaat nog niet.")
        return

    run_single_fail_research(export_csv=export_csv)

# =====================================================================
# MODE 3: HISTORICAL VIEW BACKTEST
# =====================================================================

def load_historical_backtest_data(source: str = DEFAULT_SOURCE, refresh_views: bool = True) -> pd.DataFrame:
    """Laad historische wedstrijden met odds, ECI-kansen en score uit een bronview."""
    if refresh_views:
        refresh_source_views()

    existing_relation_or_fail(source)
    available = get_table_columns(source)

    required = [
        "match_id", "date", "competition", "home_team", "away_team",
        "odds_home", "odds_draw", "odds_away",
        "home_win_pct", "draw_pct", "away_win_pct",
        "score", "home_rating", "away_rating",
    ]

    optional = [
        "result", "outcome",
        "home_drift_pct", "away_drift_pct",
        "home_drift_abs", "away_drift_abs",
        "home_range", "away_range",
        "n_snapshots", "hours_stale", "market_age_hours",
        "home_last_move_pct", "away_last_move_pct",
        "home_recent24_pct", "away_recent24_pct",
        "kickoff_at", "hours_to_kickoff", "scrape_to_kickoff_hours",
        "decision_captured_at",
    ]

    missing_required = [col for col in required if col not in available]
    if missing_required:
        raise RuntimeError(
            "Deze verplichte kolommen ontbreken in de bronview: "
            + ", ".join(missing_required)
        )

    cols = required + [col for col in optional if col in available]
    select_list = ",\n            ".join(cols)

    q = f"""
        SELECT
            {select_list}
        FROM public.{source}
        WHERE odds_home IS NOT NULL
          AND odds_draw IS NOT NULL
          AND odds_away IS NOT NULL
          AND home_win_pct IS NOT NULL
          AND draw_pct IS NOT NULL
          AND away_win_pct IS NOT NULL
          AND score IS NOT NULL
    """

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    print(f"[load] rows loaded: {len(df)}")
    return df


def parse_score_to_result_code(score) -> str | None:
    if score is None or pd.isna(score):
        return None

    extracted = pd.Series(str(score)).str.extract(r"(\d+)\D+(\d+)").iloc[0]
    if extracted.isna().any():
        return None

    home_goals = int(extracted.iloc[0])
    away_goals = int(extracted.iloc[1])

    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def normalize_result(row: pd.Series) -> str | None:
    for col in ["result", "outcome"]:
        if col not in row or pd.isna(row[col]):
            continue

        raw = str(row[col]).strip().upper()
        mapping = {
            "H": "H", "HOME": "H", "HOME_WIN": "H", "HOME WIN": "H", "1": "H",
            "A": "A", "AWAY": "A", "AWAY_WIN": "A", "AWAY WIN": "A", "2": "A",
            "D": "D", "DRAW": "D", "DRAWN": "D", "X": "D",
        }
        if raw in mapping:
            return mapping[raw]
        if "HOME" in raw and "WIN" in raw:
            return "H"
        if "AWAY" in raw and "WIN" in raw:
            return "A"
        if "DRAW" in raw:
            return "D"

    return parse_score_to_result_code(row.get("score"))


def prepare_backtest_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Maak een historische bronview geschikt voor rules.py."""
    df = df.copy()

    for col in ["home_win_pct", "draw_pct", "away_win_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not df[col].dropna().empty and df[col].dropna().max() > 1.2:
            df[col] = df[col] / 100.0

    numeric_cols = [
        "odds_home", "odds_draw", "odds_away",
        "home_rating", "away_rating",
        "home_drift_pct", "away_drift_pct",
        "home_drift_abs", "away_drift_abs",
        "home_range", "away_range",
        "hours_stale", "market_age_hours",
        "home_last_move_pct", "away_last_move_pct",
        "home_recent24_pct", "away_recent24_pct",
        "hours_to_kickoff", "scrape_to_kickoff_hours",
    ]
    ensure_columns(df, numeric_cols, np.nan)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "n_snapshots" not in df.columns:
        df["n_snapshots"] = 0
    df["n_snapshots"] = pd.to_numeric(df["n_snapshots"], errors="coerce").fillna(0).astype(int)

    df["Home Prob"] = df["home_win_pct"]
    df["Draw Prob"] = df["draw_pct"]
    df["Away Prob"] = df["away_win_pct"]

    df["bet_home"] = df["odds_home"] * df["Home Prob"]
    df["bet_draw"] = df["odds_draw"] * df["Draw Prob"]
    df["bet_away"] = df["odds_away"] * df["Away Prob"]

    df["rating_gap"] = (df["home_rating"] - df["away_rating"]).abs()
    df["rating_home_edge"] = df["home_rating"] - df["away_rating"]

    df["result_code"] = df.apply(normalize_result, axis=1)
    df = df[df["result_code"].isin(["H", "D", "A"])].copy()

    df = df[
        (df["odds_home"] > 1.01)
        & (df["odds_draw"] > 1.01)
        & (df["odds_away"] > 1.01)
        & df["Home Prob"].between(0, 1)
        & df["Draw Prob"].between(0, 1)
        & df["Away Prob"].between(0, 1)
    ].copy()

    print(f"[prepare] usable rows: {len(df)}")
    print(f"[prepare] result distribution: {df['result_code'].value_counts().to_dict()}")
    return df


def add_pick_result(picks: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return picks

    picks = picks.copy()
    picks["selected_code"] = np.select(
        [
            picks["Selection"] == "HOME",
            picks["Selection"] == "AWAY",
            picks["Selection"] == "DRAW",
        ],
        ["H", "A", "D"],
        default=None,
    )
    picks["selected_odds"] = np.select(
        [
            picks["Selection"] == "HOME",
            picks["Selection"] == "AWAY",
            picks["Selection"] == "DRAW",
        ],
        [picks["odds_home"], picks["odds_away"], picks["odds_draw"]],
        default=np.nan,
    )
    picks["won"] = picks["selected_code"] == picks["result_code"]
    picks["profit"] = np.where(picks["won"], picks["selected_odds"] - 1.0, -1.0)
    return picks


def run_main_backtest(df: pd.DataFrame) -> pd.DataFrame:
    work = apply_rules(df)
    work = apply_drift(work)
    picks = build_picks(work)

    if picks.empty:
        return picks

    if "result_code" not in picks.columns:
        picks = picks.merge(df[["match_id", "result_code"]], on="match_id", how="left")

    return add_pick_result(picks)


def build_single_fail_candidates(df_rules: pd.DataFrame) -> pd.DataFrame:
    sf = df_rules[
        (df_rules["home_fail_count"] == 1) | (df_rules["away_fail_count"] == 1)
    ].copy()

    if sf.empty:
        return sf

    sf["Selection"] = sf.apply(
        lambda row: choose_relevant_side(
            row,
            home_condition=(row.get("home_fail_count") == 1),
            away_condition=(row.get("away_fail_count") == 1),
        ),
        axis=1,
    )
    sf = sf[sf["Selection"].notna()].copy()

    sf["fail_reason"] = np.where(
        sf["Selection"] == "HOME",
        sf["home_fail_reasons"],
        sf["away_fail_reasons"],
    )
    sf["single_fail_raw_strength"] = np.where(
        sf["Selection"] == "HOME",
        sf["RawStrength_Home_All"],
        sf["RawStrength_Away_All"],
    )
    sf["selected_drift_pct"] = np.where(
        sf["Selection"] == "HOME",
        sf["home_drift_pct"],
        sf["away_drift_pct"],
    )
    sf["strength_bucket"] = sf["single_fail_raw_strength"].apply(make_strength_bucket)
    sf["PickType"] = "SINGLE_FAIL"

    return add_pick_result(sf)


def summarize_single_fails_grouped(bets: pd.DataFrame, group_cols: list[str], min_bets: int = 1) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    out = (
        bets.groupby(group_cols, observed=True)
        .agg(
            bets=("profit", "size"),
            wins=("won", "sum"),
            profit=("profit", "sum"),
            avg_odds=("selected_odds", "mean"),
            avg_strength=("single_fail_raw_strength", "mean"),
            max_strength=("single_fail_raw_strength", "max"),
        )
        .reset_index()
    )
    out["roi"] = out["profit"] / out["bets"]
    out["hitrate"] = out["wins"] / out["bets"]
    out = out[out["bets"] >= min_bets].copy()
    return out.sort_values(["roi", "bets"], ascending=[False, False])


def run_historical_backtest(source: str, refresh_views: bool = True, export_csv: bool = False) -> None:
    print_header("BETMOBILE HISTORICAL RESEARCH BACKTEST")
    print(f"source={source}")
    print(
        "rules="
        f"min_prob={RULE_MIN_PROB}, "
        f"min_value={RULE_MIN_VALUE}, "
        f"rating_gap={RULE_MIN_RATING_GAP}, "
        f"odds=[{RULE_MIN_ODDS},{RULE_MAX_ODDS}], "
        f"snapshots={RULE_MIN_SNAPSHOTS}, "
        f"min_drift_abs={RULE_MIN_DRIFT_ABS}, "
        f"min_strength={MIN_STRENGTH}"
    )

    raw = load_historical_backtest_data(source=source, refresh_views=refresh_views)
    df = prepare_backtest_frame(raw)

    if df.empty:
        print("Geen bruikbare historische data gevonden.")
        return

    picks = run_main_backtest(df)

    print_header("OVERALL HISTORICAL PICK RESULTS")
    if picks.empty:
        print("Geen picks gevonden.")
    else:
        for pick_type, part in picks.groupby("PickType", observed=True):
            print_result(summarize_bets(str(pick_type), part))
        print_result(summarize_bets("ALL HISTORICAL PICKS", picks))

    df_rules = apply_rules(df)
    single_fails = build_single_fail_candidates(df_rules)

    print_header("SINGLE FAIL RESULTS")
    print_result(summarize_bets("SINGLE_FAIL ALL", single_fails))

    by_reason = summarize_single_fails_grouped(single_fails, ["fail_reason"])
    by_reason_strength = summarize_single_fails_grouped(single_fails, ["fail_reason", "strength_bucket"])
    by_comp_reason = summarize_single_fails_grouped(single_fails, ["competition", "fail_reason"], min_bets=10)

    print_table("SINGLE_FAIL BY REASON", by_reason)
    print_table("SINGLE_FAIL BY REASON + STRENGTH", by_reason_strength)
    print_table("SINGLE_FAIL BY COMPETITION + REASON (min 10)", by_comp_reason)

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not picks.empty:
            path = EXPORT_DIR / f"historical_backtest_picks_{stamp}.csv"
            picks.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[export] {path}")
        if not single_fails.empty:
            path = EXPORT_DIR / f"historical_single_fails_{stamp}.csv"
            single_fails.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[export] {path}")


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Betmobile research backtest")
    parser.add_argument(
        "--mode",
        choices=["all", "picks", "singlefails", "historical"],
        default="all",
        help=(
            "all = picks + singlefails; "
            "picks = public.picks_evaluated; "
            "singlefails = public.picks_single_fail_candidates; "
            "historical = brede bronview backtesten"
        ),
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Bronview/tabel in public schema voor mode=historical",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Refresh materialized views niet vooraf in mode=historical",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Schrijf CSV exports naar output/research",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "all":
        run_all_research(export_csv=args.export_csv)
    elif args.mode == "picks":
        run_picks_evaluated_research(export_csv=args.export_csv)
    elif args.mode == "singlefails":
        run_single_fail_research(export_csv=args.export_csv)
    else:
        run_historical_backtest(
            source=args.source,
            refresh_views=not args.no_refresh,
            export_csv=args.export_csv,
        )
