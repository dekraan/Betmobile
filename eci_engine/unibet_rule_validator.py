r"""
unibet_rule_validator.py

Validator voor Betmobile basisregels op basis van public.eci_oddspedia_matches.

Doel
----
Niet blind autotunen op maximale ROI, maar sanity-checken of de vaste ECI basisregels
uit config.py logisch blijven op een relatief schone, vaste bookmaker-dataset:
1 seizoen Oddspedia/Unibet gekoppeld aan ECI.

Bron
----
public.eci_oddspedia_matches

Wat dit script doet
-------------------
1. Sanity checks op de brondata
2. Long-form HOME/AWAY bouwen, DRAW standaard alleen als referentie uitgesloten
3. Huidige config-regels testen
4. Sensitivity-analyse voor min_prob, min_value, odds range, rating_gap
5. Segmentanalyse op side/prob/value/odds/rating/competition
6. Robuuste kandidaat-zones vinden
7. Danger zones vinden
8. Suggested baseline rules exporteren als JSON, maar NIETS automatisch aanpassen

Gebruik
-------
Plaats in:
    C:\Users\Gebruiker\Documents\Betmobile\eci_engine\unibet_rule_validator.py

Run:
    python unibet_rule_validator.py

Optioneel:
    python unibet_rule_validator.py --min-bets-segment 100 --min-bets-rule 150

Output
------
    output/research/unibet_rule_validator/

Belangrijk
----------
Dit script verandert geen productiecode en schrijft niets naar de database.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

try:
    from db import db_engine
    from config import (
        OUTPUT_DIR,
        RULE_MIN_PROB,
        RULE_MIN_VALUE,
        RULE_MIN_RATING_GAP,
        RULE_MIN_ODDS,
        RULE_MAX_ODDS,
    )
except ImportError as e:
    raise SystemExit(
        "Kon projectmodules niet importeren. Run dit script vanuit de eci_engine map.\n"
        f"Originele fout: {e}"
    )


SOURCE_TABLE = "eci_oddspedia_matches"
EXPORT_DIR = OUTPUT_DIR / "research" / "unibet_rule_validator"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Rule:
    name: str
    side: str | None = None  # HOME, AWAY of None voor beide
    min_prob: float = RULE_MIN_PROB
    min_value: float = RULE_MIN_VALUE
    min_rating_gap: float = RULE_MIN_RATING_GAP
    min_odds: float = RULE_MIN_ODDS
    max_odds: float = RULE_MAX_ODDS
    require_home_edge: bool = True


@dataclass
class Summary:
    name: str
    bets: int
    wins: int
    losses: int
    profit: float
    roi: float
    hitrate: float
    avg_odds: float | None
    avg_prob: float | None
    avg_value: float | None
    avg_rating_gap: float | None
    first_date: str | None
    last_date: str | None


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def relation_exists(name: str) -> bool:
    q = text(
        """
        SELECT 1
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = :name
        LIMIT 1
        """
    )
    with db_engine().connect() as conn:
        return conn.execute(q, {"name": name}).fetchone() is not None


def get_columns(name: str) -> set[str]:
    q = text(
        """
        SELECT a.attname
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = :name
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """
    )
    with db_engine().connect() as conn:
        return {r[0] for r in conn.execute(q, {"name": name}).fetchall()}


def parse_score(score) -> str | None:
    if score is None or pd.isna(score):
        return None
    nums = re.findall(r"\d+", str(score))
    if len(nums) < 2:
        return None
    h, a = int(nums[0]), int(nums[1])
    if h > a:
        return "HOME"
    if a > h:
        return "AWAY"
    return "DRAW"


def normalize_result_value(result, score=None) -> str | None:
    if result is not None and not pd.isna(result):
        s = str(result).strip().upper()
        mapping = {
            "H": "HOME", "HOME": "HOME", "HOME_WIN": "HOME", "HOME WIN": "HOME", "1": "HOME",
            "A": "AWAY", "AWAY": "AWAY", "AWAY_WIN": "AWAY", "AWAY WIN": "AWAY", "2": "AWAY",
            "D": "DRAW", "X": "DRAW", "DRAW": "DRAW", "DRAWN": "DRAW",
        }
        if s in mapping:
            return mapping[s]
        if "HOME" in s and "WIN" in s:
            return "HOME"
        if "AWAY" in s and "WIN" in s:
            return "AWAY"
        if "DRAW" in s:
            return "DRAW"
    return parse_score(score)


def pct_to_fraction(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if s.dropna().quantile(0.95) > 1.2:
        return s / 100.0
    return s


def load_source() -> pd.DataFrame:
    if not relation_exists(SOURCE_TABLE):
        raise RuntimeError(f"public.{SOURCE_TABLE} bestaat niet.")

    available = get_columns(SOURCE_TABLE)
    required = [
        "match_id",
        "home_team",
        "away_team",
        "odds_home",
        "odds_draw",
        "odds_away",
        "home_win_pct",
        "draw_pct",
        "away_win_pct",
        "home_rating",
        "away_rating",
        "competition",
        "score",
        "result",
    ]
    missing = [c for c in required if c not in available]
    if missing:
        raise RuntimeError(
            "Deze verplichte kolommen ontbreken in public.eci_oddspedia_matches: "
            + ", ".join(missing)
        )

    optional = [
        "eci_date",
        "oddspedia_date",
        "date_diff_days",
        "oddspedia_home_team",
        "oddspedia_away_team",
        "implied_home",
        "implied_draw",
        "implied_away",
        "value_home",
        "value_draw",
        "value_away",
        "rating_diff",
        "oddspedia_competition",
        "status",
    ]
    cols = required + [c for c in optional if c in available]
    select_list = ",\n            ".join(cols)

    where_parts = [
        "odds_home IS NOT NULL",
        "odds_away IS NOT NULL",
        "home_win_pct IS NOT NULL",
        "away_win_pct IS NOT NULL",
        "score IS NOT NULL",
    ]
    if "status" in available:
        # Houd finished als status bestaat; als waarden anders heten, laten we later alsnog via result/score filteren.
        where_parts.append("(status IS NULL OR lower(status::text) IN ('finished','ft','complete','completed'))")

    q = f"""
        SELECT
            {select_list}
        FROM public.{SOURCE_TABLE}
        WHERE {' AND '.join(where_parts)}
    """

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    return prepare_source(df)


def prepare_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_col = "eci_date" if "eci_date" in df.columns else "oddspedia_date" if "oddspedia_date" in df.columns else None
    if date_col:
        df["date_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date_dt"] = pd.NaT

    for c in ["home_win_pct", "draw_pct", "away_win_pct"]:
        df[c] = pct_to_fraction(df[c])

    numeric_cols = [
        "odds_home", "odds_draw", "odds_away",
        "home_rating", "away_rating", "rating_diff", "date_diff_days",
        "value_home", "value_draw", "value_away",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "rating_diff" not in df.columns or df["rating_diff"].isna().all():
        df["rating_diff"] = df["home_rating"] - df["away_rating"]

    df["rating_gap"] = df["rating_diff"].abs()
    df["rating_home_edge"] = df["rating_diff"]

    df["result_norm"] = [normalize_result_value(r, s) for r, s in zip(df["result"], df["score"])]

    df = df[
        df["result_norm"].isin(["HOME", "DRAW", "AWAY"])
        & df["odds_home"].gt(1.01)
        & df["odds_away"].gt(1.01)
        & df["home_win_pct"].between(0, 1)
        & df["away_win_pct"].between(0, 1)
    ].copy()

    df["value_home_calc"] = df["home_win_pct"] * df["odds_home"]
    df["value_away_calc"] = df["away_win_pct"] * df["odds_away"]

    return df


def to_long(df: pd.DataFrame, include_draw: bool = False) -> pd.DataFrame:
    common = {
        "match_id": df["match_id"],
        "date_dt": df["date_dt"],
        "competition": df["competition"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "score": df["score"],
        "result_norm": df["result_norm"],
        "home_rating": df["home_rating"],
        "away_rating": df["away_rating"],
        "rating_diff": df["rating_diff"],
        "rating_gap": df["rating_gap"],
        "rating_home_edge": df["rating_home_edge"],
    }
    if "oddspedia_competition" in df.columns:
        common["oddspedia_competition"] = df["oddspedia_competition"]
    if "date_diff_days" in df.columns:
        common["date_diff_days"] = df["date_diff_days"]

    home = pd.DataFrame(
        {
            **common,
            "side": "HOME",
            "prob": df["home_win_pct"],
            "odds": df["odds_home"],
            "value": df["value_home_calc"],
            "selected_rating_edge": df["rating_home_edge"],
        }
    )
    away = pd.DataFrame(
        {
            **common,
            "side": "AWAY",
            "prob": df["away_win_pct"],
            "odds": df["odds_away"],
            "value": df["value_away_calc"],
            "selected_rating_edge": -df["rating_home_edge"],
        }
    )
    frames = [home, away]

    if include_draw:
        draw = pd.DataFrame(
            {
                **common,
                "side": "DRAW",
                "prob": df["draw_pct"],
                "odds": df["odds_draw"],
                "value": df["draw_pct"] * df["odds_draw"],
                "selected_rating_edge": 0.0,
            }
        )
        frames.append(draw)

    long = pd.concat(frames, ignore_index=True)
    long["won"] = long["side"] == long["result_norm"]
    long["profit"] = np.where(long["won"], long["odds"] - 1.0, -1.0)

    add_buckets(long)
    return long


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df["prob_bucket"] = pd.cut(
        df["prob"],
        bins=[0, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70, 1.0],
        labels=["<50%", "50-52%", "52-54%", "54-56%", "56-58%", "58-60%", "60-65%", "65-70%", "70%+"],
        include_lowest=True,
    )
    df["value_bucket"] = pd.cut(
        df["value"],
        bins=[0, 0.95, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.25, np.inf],
        labels=["<0.95", "0.95-1.00", "1.00-1.02", "1.02-1.04", "1.04-1.06", "1.06-1.08", "1.08-1.10", "1.10-1.15", "1.15-1.25", "1.25+"],
        include_lowest=True,
    )
    df["odds_bucket"] = pd.cut(
        df["odds"],
        bins=[1.0, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 10.0, np.inf],
        labels=["1.0-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.5", "2.5-3.0", "3.0-4.0", "4.0-10", "10+"],
        include_lowest=True,
    )
    df["rating_gap_bucket"] = pd.cut(
        df["rating_gap"],
        bins=[-0.1, 100, 250, 500, 750, 1000, 1500, 2500, np.inf],
        labels=["0-100", "100-250", "250-500", "500-750", "750-1000", "1000-1500", "1500-2500", "2500+"],
    )
    df["favorite_type"] = np.select(
        [
            df["side"].eq("HOME") & df["rating_home_edge"].gt(0),
            df["side"].eq("HOME") & df["rating_home_edge"].lt(0),
            df["side"].eq("AWAY") & df["rating_home_edge"].lt(0),
            df["side"].eq("AWAY") & df["rating_home_edge"].gt(0),
        ],
        ["HOME_FAVORITE", "HOME_UNDERDOG", "AWAY_FAVORITE", "AWAY_UNDERDOG"],
        default="EVEN_RATING",
    )
    return df


def summarize(name: str, bets: pd.DataFrame) -> Summary:
    if bets is None or bets.empty:
        return Summary(name, 0, 0, 0, 0.0, 0.0, 0.0, None, None, None, None, None, None)

    n = int(len(bets))
    wins = int(bets["won"].sum())
    profit = float(bets["profit"].sum())
    dates = pd.to_datetime(bets["date_dt"], errors="coerce")
    first = dates.min()
    last = dates.max()
    return Summary(
        name=name,
        bets=n,
        wins=wins,
        losses=n - wins,
        profit=profit,
        roi=profit / n if n else 0.0,
        hitrate=wins / n if n else 0.0,
        avg_odds=float(bets["odds"].mean()) if n else None,
        avg_prob=float(bets["prob"].mean()) if n else None,
        avg_value=float(bets["value"].mean()) if n else None,
        avg_rating_gap=float(bets["rating_gap"].mean()) if n else None,
        first_date=first.date().isoformat() if pd.notna(first) else None,
        last_date=last.date().isoformat() if pd.notna(last) else None,
    )


def summaries_to_frame(rows: list[Summary]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    if df.empty:
        return df
    for c in ["profit", "roi", "hitrate", "avg_odds", "avg_prob", "avg_value", "avg_rating_gap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def select_rule(long: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    mask = (
        long["prob"].ge(rule.min_prob)
        & long["value"].ge(rule.min_value)
        & long["rating_gap"].ge(rule.min_rating_gap)
        & long["odds"].between(rule.min_odds, rule.max_odds, inclusive="both")
    )
    if rule.side:
        mask &= long["side"].eq(rule.side)
    if rule.require_home_edge:
        # Zelfde principe als productie: HOME alleen als home rating edge positief is.
        # AWAY mag ook als away lager rated is; dat is historisch bewust zo in huidige rules.py.
        mask &= ~((long["side"].eq("HOME")) & (long["rating_home_edge"] < 0))
    return long.loc[mask].copy()


def group_summary(df: pd.DataFrame, group_cols: list[str], min_bets: int = 1) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(group_cols, observed=True, dropna=False)
        .agg(
            bets=("profit", "size"),
            wins=("won", "sum"),
            profit=("profit", "sum"),
            avg_odds=("odds", "mean"),
            avg_prob=("prob", "mean"),
            avg_value=("value", "mean"),
            avg_rating_gap=("rating_gap", "mean"),
        )
        .reset_index()
    )
    out["losses"] = out["bets"] - out["wins"]
    out["roi"] = out["profit"] / out["bets"]
    out["hitrate"] = out["wins"] / out["bets"]
    out = out[out["bets"] >= min_bets].copy()
    return out.sort_values(["roi", "bets"], ascending=[False, False])


def add_time_stability(bets: pd.DataFrame, n_slices: int = 4) -> dict:
    if bets.empty:
        return {"slice_count": 0, "min_slice_roi": 0.0, "max_slice_roi": 0.0, "positive_slice_share": 0.0}
    work = bets.sort_values("date_dt").reset_index(drop=True).copy()
    idx = np.linspace(0, len(work), n_slices + 1, dtype=int)
    rois = []
    for i in range(n_slices):
        part = work.iloc[idx[i]:idx[i + 1]]
        if len(part) < 20:
            continue
        rois.append(float(part["profit"].sum() / len(part)))
    if not rois:
        return {"slice_count": 0, "min_slice_roi": 0.0, "max_slice_roi": 0.0, "positive_slice_share": 0.0}
    return {
        "slice_count": len(rois),
        "min_slice_roi": min(rois),
        "max_slice_roi": max(rois),
        "positive_slice_share": sum(r > 0 for r in rois) / len(rois),
    }


def evaluate_rules(long: pd.DataFrame, rules: Iterable[Rule]) -> pd.DataFrame:
    rows = []
    for rule in rules:
        bets = select_rule(long, rule)
        row = asdict(summarize(rule.name, bets))
        row.update({
            "side_filter": rule.side or "BOTH",
            "min_prob": rule.min_prob,
            "min_value": rule.min_value,
            "min_rating_gap": rule.min_rating_gap,
            "min_odds": rule.min_odds,
            "max_odds": rule.max_odds,
            "require_home_edge": rule.require_home_edge,
        })
        row.update(add_time_stability(bets))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["roi", "bets"], ascending=[False, False])
    return out


def build_sanity_tables(df: pd.DataFrame, long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    tables["sanity_overall"] = pd.DataFrame([
        {"metric": "source_rows", "value": len(df)},
        {"metric": "long_rows_home_away", "value": len(long)},
        {"metric": "first_date", "value": pd.to_datetime(df["date_dt"], errors="coerce").min()},
        {"metric": "last_date", "value": pd.to_datetime(df["date_dt"], errors="coerce").max()},
        {"metric": "duplicate_match_id", "value": int(df["match_id"].duplicated().sum())},
        {"metric": "null_result_norm", "value": int(df["result_norm"].isna().sum())},
    ])

    null_cols = [
        "odds_home", "odds_draw", "odds_away", "home_win_pct", "draw_pct", "away_win_pct",
        "home_rating", "away_rating", "rating_diff", "result_norm", "score",
    ]
    null_cols = [c for c in null_cols if c in df.columns]
    tables["sanity_nulls"] = pd.DataFrame({"column": null_cols, "nulls": [int(df[c].isna().sum()) for c in null_cols]})

    tables["coverage_by_competition"] = (
        df.groupby("competition", observed=True)
        .agg(
            matches=("match_id", "size"),
            first_date=("date_dt", "min"),
            last_date=("date_dt", "max"),
            home_win_rate=("result_norm", lambda s: (s == "HOME").mean()),
            draw_rate=("result_norm", lambda s: (s == "DRAW").mean()),
            away_win_rate=("result_norm", lambda s: (s == "AWAY").mean()),
            avg_home_odds=("odds_home", "mean"),
            avg_away_odds=("odds_away", "mean"),
        )
        .reset_index()
        .sort_values("matches", ascending=False)
    )

    if "date_diff_days" in df.columns:
        tables["date_diff_distribution"] = (
            df.groupby("date_diff_days", dropna=False)
            .size()
            .reset_index(name="matches")
            .sort_values("date_diff_days")
        )

    extreme = long[
        long["odds"].ge(8)
        | long["value"].ge(1.50)
        | long["prob"].ge(0.85)
        | long["prob"].le(0.05)
    ].copy()
    tables["extreme_records"] = extreme.sort_values(["value", "odds"], ascending=[False, False]).head(300)

    duplicates = df[df["match_id"].duplicated(keep=False)].sort_values(["match_id", "date_dt"])
    tables["duplicate_match_ids"] = duplicates
    return tables


def build_sensitivity(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}

    prob_rules = []
    for side in [None, "HOME", "AWAY"]:
        for p in [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]:
            prob_rules.append(Rule(name=f"prob_{side or 'BOTH'}_{p:.2f}", side=side, min_prob=p))
    tables["sensitivity_min_prob"] = evaluate_rules(long, prob_rules).sort_values(["side_filter", "min_prob"])

    value_rules = []
    for side in [None, "HOME", "AWAY"]:
        for v in [0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20]:
            value_rules.append(Rule(name=f"value_{side or 'BOTH'}_{v:.2f}", side=side, min_value=v))
    tables["sensitivity_min_value"] = evaluate_rules(long, value_rules).sort_values(["side_filter", "min_value"])

    rating_rules = []
    for side in [None, "HOME", "AWAY"]:
        for g in [0, 100, 250, 500, 750, 1000, 1500]:
            rating_rules.append(Rule(name=f"rating_{side or 'BOTH'}_{g}", side=side, min_rating_gap=g))
    tables["sensitivity_min_rating_gap"] = evaluate_rules(long, rating_rules).sort_values(["side_filter", "min_rating_gap"])

    odds_rules = []
    odds_ranges = [(1.30, 4.00), (1.40, 4.00), (1.40, 3.00), (1.40, 2.50), (1.40, 2.20), (1.60, 2.50), (1.60, 2.20), (1.80, 2.50)]
    for side in [None, "HOME", "AWAY"]:
        for lo, hi in odds_ranges:
            odds_rules.append(Rule(name=f"odds_{side or 'BOTH'}_{lo:.2f}_{hi:.2f}", side=side, min_odds=lo, max_odds=hi))
    tables["sensitivity_odds_range"] = evaluate_rules(long, odds_rules).sort_values(["side_filter", "min_odds", "max_odds"])

    matrix_rules = []
    for side in [None, "HOME", "AWAY"]:
        for p in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
            for v in [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12]:
                matrix_rules.append(Rule(name=f"matrix_{side or 'BOTH'}_p{p:.2f}_v{v:.2f}", side=side, min_prob=p, min_value=v))
    tables["matrix_prob_value"] = evaluate_rules(long, matrix_rules)

    return tables


def build_segments(long: pd.DataFrame, min_bets_segment: int) -> dict[str, pd.DataFrame]:
    tables = {}

    base = select_rule(long, Rule(name="current_config"))
    tables["segments_current_by_side"] = group_summary(base, ["side"], min_bets=1)
    tables["segments_current_by_side_prob"] = group_summary(base, ["side", "prob_bucket"], min_bets=min_bets_segment)
    tables["segments_current_by_side_value"] = group_summary(base, ["side", "value_bucket"], min_bets=min_bets_segment)
    tables["segments_current_by_side_odds"] = group_summary(base, ["side", "odds_bucket"], min_bets=min_bets_segment)
    tables["segments_current_by_side_rating"] = group_summary(base, ["side", "rating_gap_bucket"], min_bets=min_bets_segment)
    tables["segments_current_by_competition"] = group_summary(base, ["competition"], min_bets=max(20, min_bets_segment // 2))
    tables["segments_current_by_side_favorite"] = group_summary(base, ["side", "favorite_type"], min_bets=min_bets_segment)

    # Brede segmenten op alle HOME/AWAY opties, niet alleen huidige config.
    tables["segments_all_side_prob_odds"] = group_summary(long, ["side", "prob_bucket", "odds_bucket"], min_bets=min_bets_segment)
    tables["segments_all_side_prob_value"] = group_summary(long, ["side", "prob_bucket", "value_bucket"], min_bets=min_bets_segment)
    tables["segments_all_side_value_odds"] = group_summary(long, ["side", "value_bucket", "odds_bucket"], min_bets=min_bets_segment)
    tables["segments_all_side_rating_odds"] = group_summary(long, ["side", "rating_gap_bucket", "odds_bucket"], min_bets=min_bets_segment)
    tables["segments_all_side_competition"] = group_summary(long, ["side", "competition"], min_bets=max(20, min_bets_segment // 2))

    # Compacte best/worst-lijst uit alle segmenttabellen
    rows = []
    for name, table in tables.items():
        if table is None or table.empty or "roi" not in table.columns:
            continue
        for _, r in table.iterrows():
            dims = [c for c in table.columns if c not in {
                "bets", "wins", "losses", "profit", "roi", "hitrate",
                "avg_odds", "avg_prob", "avg_value", "avg_rating_gap"
            }]
            rows.append({
                "source_table": name,
                "segment": " | ".join(f"{c}={r.get(c)}" for c in dims),
                "bets": int(r["bets"]),
                "wins": int(r["wins"]),
                "profit": float(r["profit"]),
                "roi": float(r["roi"]),
                "hitrate": float(r["hitrate"]),
                "avg_odds": float(r["avg_odds"]),
                "avg_prob": float(r["avg_prob"]),
                "avg_value": float(r["avg_value"]),
            })
    allseg = pd.DataFrame(rows)
    if not allseg.empty:
        tables["healthy_zones_top"] = allseg[allseg["bets"] >= min_bets_segment].sort_values(["roi", "bets"], ascending=[False, False]).head(100)
        tables["danger_zones_top"] = allseg[(allseg["bets"] >= min_bets_segment) & (allseg["roi"] < -0.05)].sort_values(["roi", "bets"], ascending=[True, False]).head(100)
        tables["all_segments_flat"] = allseg.sort_values(["roi", "bets"], ascending=[False, False])
    return tables


def build_candidate_rules(long: pd.DataFrame, min_bets_rule: int) -> pd.DataFrame:
    rules = []

    # Bewust grove grid: zoekt plateaus, geen pseudo-exact optimum.
    prob_grid = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
    value_grid = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12]
    rating_grid = [0, 100, 250, 500, 750, 1000]
    odds_grid = [(1.30, 4.00), (1.40, 4.00), (1.40, 3.00), (1.40, 2.50), (1.40, 2.20), (1.60, 2.50), (1.60, 2.20), (1.80, 2.50)]

    for side in ["HOME", "AWAY"]:
        for p in prob_grid:
            for v in value_grid:
                for g in rating_grid:
                    for lo, hi in odds_grid:
                        rules.append(Rule(
                            name=f"candidate_{side}_p{p:.2f}_v{v:.2f}_g{g}_o{lo:.2f}_{hi:.2f}",
                            side=side,
                            min_prob=p,
                            min_value=v,
                            min_rating_gap=g,
                            min_odds=lo,
                            max_odds=hi,
                        ))

    out = evaluate_rules(long, rules)
    if out.empty:
        return out
    out = out[out["bets"] >= min_bets_rule].copy()
    if out.empty:
        return out

    # Robuustheidsscore: geen hoogste ROI-fetish.
    # Straf zeer kleine aantallen, slechte tijdslice en te scherpe slice-instabiliteit.
    out["volume_score"] = np.minimum(out["bets"] / 500.0, 1.0)
    out["stability_score"] = out["min_slice_roi"] + (out["positive_slice_share"] * 0.05)
    out["robust_score"] = (
        out["roi"]
        + 0.35 * out["min_slice_roi"]
        + 0.10 * out["positive_slice_share"]
        + 0.05 * out["volume_score"]
    )

    # Basis: liever niet extreme filters. Daarom ook apart sorteren op robust_score.
    return out.sort_values(["robust_score", "roi", "bets"], ascending=[False, False, False])


def build_recommendation_json(current: pd.DataFrame, candidates: pd.DataFrame, danger: pd.DataFrame, args) -> dict:
    current_rows = current.to_dict(orient="records") if current is not None and not current.empty else []
    top_candidates = candidates.head(25).to_dict(orient="records") if candidates is not None and not candidates.empty else []
    danger_rows = danger.head(25).to_dict(orient="records") if danger is not None and not danger.empty else []

    verdict = "unknown"
    notes = []
    if current_rows:
        both = next((r for r in current_rows if r.get("name") == "CURRENT_CONFIG_BOTH"), None)
        if both:
            roi = float(both.get("roi", 0) or 0)
            bets = int(both.get("bets", 0) or 0)
            min_slice = float(both.get("min_slice_roi", 0) or 0)
            if bets < args.min_bets_rule:
                verdict = "too_little_volume"
                notes.append("Huidige config heeft te weinig volume voor een stevig oordeel.")
            elif roi > 0.02 and min_slice > -0.05:
                verdict = "current_config_reasonable"
                notes.append("Huidige config lijkt redelijk op deze Unibet benchmark, maar controleer side/segmenten.")
            elif roi >= -0.02:
                verdict = "current_config_neutral"
                notes.append("Huidige config is ongeveer break-even; gebruik vooral segmenten/danger-zones.")
            else:
                verdict = "current_config_weak"
                notes.append("Huidige config wordt niet goed ondersteund door deze Unibet benchmark.")

    if top_candidates:
        notes.append("Top candidates zijn research-suggesties, geen productieparameters.")
    if danger_rows:
        notes.append("Danger zones met voldoende volume verdienen eerder avoid-labels dan harde deletions.")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": f"public.{SOURCE_TABLE}",
        "purpose": "Validate fixed Betmobile ECI config against one-season Unibet/Oddspedia benchmark.",
        "verdict": verdict,
        "notes": notes,
        "current_config": {
            "min_prob": RULE_MIN_PROB,
            "min_value": RULE_MIN_VALUE,
            "min_rating_gap": RULE_MIN_RATING_GAP,
            "min_odds": RULE_MIN_ODDS,
            "max_odds": RULE_MAX_ODDS,
        },
        "current_results": current_rows,
        "suggested_candidate_rules_top25": top_candidates,
        "danger_zones_top25": danger_rows,
    }


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        if table is None:
            continue
        path = EXPORT_DIR / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[export] {path}")


def print_compact_table(title: str, df: pd.DataFrame, rows: int = 20) -> None:
    print_header(title)
    if df is None or df.empty:
        print("Geen data.")
        return
    view = df.head(rows).copy()
    for c in ["profit", "roi", "hitrate", "avg_odds", "avg_prob", "avg_value", "avg_rating_gap", "min_slice_roi", "robust_score"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(4)
    print(view.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Betmobile basisregels op Unibet/Oddspedia benchmark.")
    parser.add_argument("--min-bets-segment", type=int, default=100, help="Minimum bets voor segmenttabellen.")
    parser.add_argument("--min-bets-rule", type=int, default=150, help="Minimum bets voor kandidaatregels.")
    parser.add_argument("--include-draw", action="store_true", help="Neem DRAW ook mee in long-form referentie. Niet gebruikt voor candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 200)

    print_header("UNIBET RULE VALIDATOR")
    print(f"source=public.{SOURCE_TABLE}")
    print(f"export_dir={EXPORT_DIR}")

    df = load_source()
    long = to_long(df, include_draw=args.include_draw)
    long_home_away = long[long["side"].isin(["HOME", "AWAY"])].copy()

    print_header("LOADED")
    print(f"matches={len(df)}")
    print(f"long_home_away_rows={len(long_home_away)}")
    print(f"result_distribution={df['result_norm'].value_counts().to_dict()}")

    current_rules = [
        Rule(name="CURRENT_CONFIG_BOTH", side=None),
        Rule(name="CURRENT_CONFIG_HOME", side="HOME"),
        Rule(name="CURRENT_CONFIG_AWAY", side="AWAY"),
    ]
    current = evaluate_rules(long_home_away, current_rules)

    sanity = build_sanity_tables(df, long_home_away)
    sensitivity = build_sensitivity(long_home_away)
    segments = build_segments(long_home_away, min_bets_segment=args.min_bets_segment)
    candidates = build_candidate_rules(long_home_away, min_bets_rule=args.min_bets_rule)

    danger = segments.get("danger_zones_top", pd.DataFrame())
    recommendation = build_recommendation_json(current, candidates, danger, args)

    all_tables = {
        **sanity,
        "current_config_results": current,
        **sensitivity,
        **segments,
        "candidate_rules_ranked": candidates,
    }
    save_tables(all_tables)

    rec_path = EXPORT_DIR / "recommendation_summary.json"
    rec_path.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[export] {rec_path}")

    print_compact_table("CURRENT CONFIG", current)
    print_compact_table("TOP CANDIDATE RULES BY ROBUST SCORE", candidates, rows=25)
    print_compact_table("HEALTHY ZONES", segments.get("healthy_zones_top", pd.DataFrame()), rows=25)
    print_compact_table("DANGER ZONES", danger, rows=25)

    print_header("VERDICT")
    print(recommendation["verdict"])
    for note in recommendation["notes"]:
        print(f"- {note}")


if __name__ == "__main__":
    main()
