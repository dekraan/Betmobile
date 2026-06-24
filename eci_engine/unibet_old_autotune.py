r"""
unibet_old_autotune.py

Quick & dirty oude-autotuner-run op de nieuwe fixed Oddspedia/Unibet benchmark:
    public.eci_oddspedia_matches

Doel
----
Deze doet bewust iets anders dan unibet_rule_validator.py.

- unibet_rule_validator.py = sanity/validator/robuuste zones.
- dit script = oude autotune-logica zo eerlijk mogelijk opnieuw draaien op de nieuwe Unibet-set.

Omdat public.eci_oddspedia_matches géén Bet365 snapshot/drift-history bevat,
worden drift- en snapshotfilters NIET meegenomen in de grid. Dat is expres.
Anders zou je schijnprecisie krijgen.

Wat wordt wel getuned?
----------------------
- side: BOTH / HOME / AWAY
- min_prob
- min_value = probability * odds
- min_rating_gap = abs(home_rating - away_rating)
- min_odds
- max_odds
- min_strength, berekend met dezelfde simpele formule uit de oude autotuner:

    raw_strength =
        max(value - min_value, 0) * 2
      + max(prob - min_prob, 0) * 2
      + rating_gap / 500
      + HOME bonus: max(rating_home_edge, 0) / 500

Belangrijk
----------
Dit script past niets aan in config.py, rules.py of eci_picks.py.
Het schrijft alleen CSV/JSON naar:
    output/research/unibet_old_autotune/

Plaats in:
    C:\Users\Gebruiker\Documents\Betmobile\eci_engine\unibet_old_autotune.py

Run:
    python unibet_old_autotune.py

Optioneel:
    python unibet_old_autotune.py --min-bets 300
    python unibet_old_autotune.py --include-current-config
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

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
        MIN_STRENGTH,
    )
except ImportError as e:
    raise SystemExit(
        "Kon projectmodules niet importeren. Run dit script vanuit de eci_engine-map.\n"
        f"Originele fout: {e}"
    )

SOURCE_TABLE = "eci_oddspedia_matches"
EXPORT_DIR = OUTPUT_DIR / "research" / "unibet_old_autotune"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

N_SLICES_STABILITY = 4
STABILITY_WEIGHT = 0.5


@dataclass(frozen=True)
class Params:
    side: str  # BOTH, HOME, AWAY
    min_prob: float
    min_value: float
    min_rating_gap: int
    min_odds: float
    max_odds: float
    min_strength: float
    require_home_edge: bool = True


def relation_exists(name: str) -> bool:
    q = text("""
        SELECT 1
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = :name
        LIMIT 1
    """)
    with db_engine().connect() as conn:
        return conn.execute(q, {"name": name}).fetchone() is not None


def get_columns(name: str) -> set[str]:
    q = text("""
        SELECT a.attname
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = :name
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
    """)
    with db_engine().connect() as conn:
        return {r[0] for r in conn.execute(q, {"name": name}).fetchall()}


def parse_score(score) -> Optional[str]:
    if score is None or pd.isna(score):
        return None
    nums = re.findall(r"\d+", str(score))
    if len(nums) < 2:
        return None
    home_goals, away_goals = int(nums[0]), int(nums[1])
    if home_goals > away_goals:
        return "HOME"
    if away_goals > home_goals:
        return "AWAY"
    return "DRAW"


def normalize_result(result, score=None) -> Optional[str]:
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


def pct_to_fraction(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if not out.dropna().empty and out.dropna().quantile(0.95) > 1.2:
        out = out / 100.0
    return out


def load_source() -> pd.DataFrame:
    if not relation_exists(SOURCE_TABLE):
        raise RuntimeError(f"public.{SOURCE_TABLE} bestaat niet.")

    available = get_columns(SOURCE_TABLE)
    required = [
        "match_id", "home_team", "away_team", "competition",
        "odds_home", "odds_draw", "odds_away",
        "home_win_pct", "draw_pct", "away_win_pct",
        "home_rating", "away_rating", "score", "result",
    ]
    missing = [c for c in required if c not in available]
    if missing:
        raise RuntimeError("Ontbrekende kolommen in public.eci_oddspedia_matches: " + ", ".join(missing))

    optional = ["eci_date", "oddspedia_date", "date_diff_days", "rating_diff", "status", "oddspedia_competition"]
    cols = required + [c for c in optional if c in available]
    select_list = ",\n            ".join(cols)

    where = [
        "odds_home IS NOT NULL",
        "odds_draw IS NOT NULL",
        "odds_away IS NOT NULL",
        "home_win_pct IS NOT NULL",
        "away_win_pct IS NOT NULL",
        "score IS NOT NULL",
    ]
    if "status" in available:
        where.append("(status IS NULL OR lower(status::text) IN ('finished','ft','complete','completed'))")

    q = f"""
        SELECT
            {select_list}
        FROM public.{SOURCE_TABLE}
        WHERE {' AND '.join(where)}
    """

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    date_col = "eci_date" if "eci_date" in df.columns else "oddspedia_date" if "oddspedia_date" in df.columns else None
    df["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT

    for c in ["home_win_pct", "draw_pct", "away_win_pct"]:
        df[c] = pct_to_fraction(df[c])

    for c in ["odds_home", "odds_draw", "odds_away", "home_rating", "away_rating", "rating_diff"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "rating_diff" not in df.columns or df["rating_diff"].isna().all():
        df["rating_diff"] = df["home_rating"] - df["away_rating"]

    df["result_norm"] = [normalize_result(r, s) for r, s in zip(df["result"], df["score"])]

    df = df[
        df["result_norm"].isin(["HOME", "DRAW", "AWAY"])
        & df["odds_home"].gt(1.01)
        & df["odds_away"].gt(1.01)
        & df["home_win_pct"].between(0, 1)
        & df["away_win_pct"].between(0, 1)
    ].copy()

    return df


def to_long(df: pd.DataFrame, include_draw: bool = False) -> pd.DataFrame:
    rating_gap = df["rating_diff"].abs()
    rating_home_edge = df["rating_diff"]

    common = {
        "match_id": df["match_id"],
        "date": df["date"],
        "competition": df["competition"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_rating": df["home_rating"],
        "away_rating": df["away_rating"],
        "rating_gap": rating_gap,
        "rating_home_edge": rating_home_edge,
        "score": df["score"],
        "result_norm": df["result_norm"],
    }

    home = pd.DataFrame({
        **common,
        "side": "HOME",
        "prob": df["home_win_pct"].astype(float),
        "odds": df["odds_home"].astype(float),
        "edge_ok": rating_home_edge >= 0,
    })
    away = pd.DataFrame({
        **common,
        "side": "AWAY",
        "prob": df["away_win_pct"].astype(float),
        "odds": df["odds_away"].astype(float),
        # Zelfde oude/current principe: AWAY hoeft niet per se rating-favoriet te zijn.
        "edge_ok": True,
    })

    frames = [home, away]
    if include_draw:
        draw = pd.DataFrame({
            **common,
            "side": "DRAW",
            "prob": df["draw_pct"].astype(float),
            "odds": df["odds_draw"].astype(float),
            "edge_ok": True,
        })
        frames.append(draw)

    long = pd.concat(frames, ignore_index=True)
    long["value"] = long["prob"] * long["odds"]
    long["won"] = long["side"] == long["result_norm"]
    long["profit"] = np.where(long["won"], long["odds"] - 1.0, -1.0)
    return long


def select_bets(long: pd.DataFrame, params: Params) -> pd.DataFrame:
    mask = (
        long["prob"].ge(params.min_prob)
        & long["value"].ge(params.min_value)
        & long["rating_gap"].ge(params.min_rating_gap)
        & long["odds"].between(params.min_odds, params.max_odds, inclusive="both")
    )
    if params.side != "BOTH":
        mask &= long["side"].eq(params.side)
    if params.require_home_edge:
        mask &= ~((long["side"].eq("HOME")) & (~long["edge_ok"]))

    bets = long.loc[mask].copy()
    if bets.empty:
        return bets

    bets["raw_strength"] = (
        (bets["value"] - params.min_value).clip(lower=0) * 2
        + (bets["prob"] - params.min_prob).clip(lower=0) * 2
        + bets["rating_gap"].clip(lower=0) / 500
    )
    bets["raw_strength"] += np.where(
        bets["side"].eq("HOME"),
        bets["rating_home_edge"].clip(lower=0) / 500,
        0.0,
    )
    bets = bets[bets["raw_strength"] >= params.min_strength].copy()
    return bets


def basic_stats(bets: pd.DataFrame) -> dict:
    n = len(bets)
    if n == 0:
        return {"bets": 0, "wins": 0, "profit": 0.0, "roi": 0.0, "hitrate": 0.0, "avg_odds": None, "avg_prob": None, "avg_value": None}
    wins = int(bets["won"].sum())
    profit = float(bets["profit"].sum())
    return {
        "bets": int(n),
        "wins": wins,
        "profit": profit,
        "roi": profit / n,
        "hitrate": wins / n,
        "avg_odds": float(bets["odds"].mean()),
        "avg_prob": float(bets["prob"].mean()),
        "avg_value": float(bets["value"].mean()),
        "avg_strength": float(bets["raw_strength"].mean()) if "raw_strength" in bets.columns else None,
        "first_date": pd.to_datetime(bets["date"], errors="coerce").min(),
        "last_date": pd.to_datetime(bets["date"], errors="coerce").max(),
    }


def stability(bets: pd.DataFrame, n_slices: int = N_SLICES_STABILITY) -> dict:
    if bets.empty:
        return {"slice_count": 0, "min_slice_roi": 0.0, "max_slice_roi": 0.0, "positive_slice_share": 0.0}
    work = bets.sort_values("date").reset_index(drop=True)
    idx = np.linspace(0, len(work), n_slices + 1, dtype=int)
    rois = []
    for i in range(n_slices):
        part = work.iloc[idx[i]:idx[i + 1]]
        if len(part) < 30:
            continue
        rois.append(float(part["profit"].sum() / len(part)))
    if not rois:
        return {"slice_count": 0, "min_slice_roi": 0.0, "max_slice_roi": 0.0, "positive_slice_share": 0.0}
    return {
        "slice_count": len(rois),
        "min_slice_roi": min(rois),
        "max_slice_roi": max(rois),
        "positive_slice_share": sum(x > 0 for x in rois) / len(rois),
    }


def evaluate(long: pd.DataFrame, params: Params) -> dict:
    bets = select_bets(long, params)
    row = asdict(params)
    row.update(basic_stats(bets))
    row.update(stability(bets))
    row["score"] = row["roi"] + STABILITY_WEIGHT * row["min_slice_roi"]
    return row


def make_grid(args) -> list[Params]:
    sides = ["BOTH", "HOME", "AWAY"] if args.include_side_grid else ["BOTH"]

    # Dit is bewust dicht bij de oude grid, maar zonder snapshots/drift.
    # Extra waarden rond de huidige config zijn toegevoegd om 0.52/1.04 eerlijk te testen.
    prob_grid = [0.22, 0.28, 0.34, 0.40, 0.46, 0.50, 0.52, 0.54, 0.58, 0.60, 0.62]
    value_grid = [1.00, 1.02, 1.04, 1.08, 1.10, 1.12, 1.16]
    rating_grid = [0, 100, 200, 250, 400, 500, 750, 1000]
    min_odds_grid = [1.30, 1.40, 1.60, 1.80]
    max_odds_grid = [2.20, 2.50, 3.00, 4.00, 5.00, 6.00]
    strength_grid = [0.0, 1.5, 2.0, 2.5]

    params = []
    for side, p, v, g, lo, hi, st in product(
        sides,
        prob_grid,
        value_grid,
        rating_grid,
        min_odds_grid,
        max_odds_grid,
        strength_grid,
    ):
        if hi <= lo + 0.01:
            continue
        params.append(Params(side=side, min_prob=p, min_value=v, min_rating_gap=g, min_odds=lo, max_odds=hi, min_strength=st))
    return params


def current_config_rows(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["BOTH", "HOME", "AWAY"]:
        params = Params(
            side=side,
            min_prob=float(RULE_MIN_PROB),
            min_value=float(RULE_MIN_VALUE),
            min_rating_gap=int(RULE_MIN_RATING_GAP),
            min_odds=float(RULE_MIN_ODDS),
            max_odds=float(RULE_MAX_ODDS),
            min_strength=float(MIN_STRENGTH),
        )
        rows.append(evaluate(long, params))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Oude autotuner quick & dirty op public.eci_oddspedia_matches.")
    parser.add_argument("--min-bets", type=int, default=300, help="Minimum aantal bets voor gridresultaten.")
    parser.add_argument("--include-side-grid", action="store_true", help="Tune ook HOME/AWAY apart. Standaard alleen BOTH, zoals oude config.")
    parser.add_argument("--include-draw", action="store_true", help="Neem DRAW mee in long-form. Standaard uit.")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 80)

    print("=== UNIBET OLD AUTOTUNE ===")
    print(f"source=public.{SOURCE_TABLE}")
    print(f"export_dir={EXPORT_DIR}")
    print(f"min_bets={args.min_bets}")

    df = load_source()
    long = to_long(df, include_draw=args.include_draw)
    if not args.include_draw:
        long = long[long["side"].isin(["HOME", "AWAY"])].copy()

    print(f"matches={len(df)}")
    print(f"long_rows={len(long)}")
    print("result_distribution=", df["result_norm"].value_counts().to_dict())

    current = current_config_rows(long)
    current.to_csv(EXPORT_DIR / "current_config_with_old_strength.csv", index=False, encoding="utf-8-sig")

    grid = make_grid(args)
    print(f"grid_combinations={len(grid)}")

    rows = []
    for i, params in enumerate(grid, start=1):
        if i % 500 == 0 or i == len(grid):
            print(f"[grid] {i}/{len(grid)}")
        row = evaluate(long, params)
        if row["bets"] >= args.min_bets:
            rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        print("Geen gridresultaten boven min_bets.")
        return

    results = results.sort_values(["score", "roi", "min_slice_roi", "profit", "bets"], ascending=[False, False, False, False, False]).copy()
    results.to_csv(EXPORT_DIR / "old_autotune_gridsearch.csv", index=False, encoding="utf-8-sig")

    best = results.iloc[0].to_dict()
    out_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": f"public.{SOURCE_TABLE}",
        "note": "Old autotune style on fixed Unibet/Oddspedia data. Drift/snapshots unavailable and therefore excluded.",
        "min_bets": args.min_bets,
        "best": {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in best.items()},
        "current_config_with_old_strength": current.to_dict(orient="records"),
    }
    (EXPORT_DIR / "old_autotune_best.json").write_text(json.dumps(out_json, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== CURRENT CONFIG WITH OLD STRENGTH ===")
    print(current[["side", "bets", "wins", "profit", "roi", "hitrate", "avg_odds", "avg_prob", "avg_value", "avg_strength", "min_slice_roi", "score"]].to_string(index=False))

    print("\n=== TOP 25 OLD-AUTOTUNE RESULTS ===")
    show_cols = [
        "side", "min_prob", "min_value", "min_rating_gap", "min_odds", "max_odds", "min_strength",
        "bets", "wins", "profit", "roi", "hitrate", "avg_odds", "avg_prob", "avg_value", "avg_strength",
        "min_slice_roi", "max_slice_roi", "positive_slice_share", "score",
    ]
    print(results[show_cols].head(25).to_string(index=False))

    print("\n[export]", EXPORT_DIR / "current_config_with_old_strength.csv")
    print("[export]", EXPORT_DIR / "old_autotune_gridsearch.csv")
    print("[export]", EXPORT_DIR / "old_autotune_best.json")


if __name__ == "__main__":
    main()
