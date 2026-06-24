"""
autotune_eci_rules.py

Automatische tuning van ECI-rules op basis van historische data in 'betmobile' (DB 'Betmobile').

- Haalt data op uit PostgreSQL (table/view 'betmobile')
- Maakt long-form (1 rij per match+outcome)
- Doet een gridsearch over:
    * min_prob
    * min_value (prob * odds)
    * min_rating_gap (absolute rating-diff)
    * min_odds
    * max_odds
    * min_snapshots (n_snapshots)
    * min_drift_abs (absolute oddsdrift van richting waarop je inzet)
- Evalueert iedere combinatie op:
    * overall ROI
    * min ROI over 4 tijd-slices
    * volume (minimaal aantal bets)
- Schrijft resultaten naar CSV en slaat beste combo op als JSON.
- Probeert optioneel ECI_RULE_PARAMS in eci_picks.py bij te werken.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ========== DB CONFIG ==========
DB_USER = "postgres"
DB_PASS = "300500"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "Betmobile"

# ========== AUTO-TUNE SETTINGS ==========

# Minimale volume-eis voor een parametercombinatie
MIN_BETS_GLOBAL = 300

# Aantal tijd-slices voor stabiliteitscheck
N_SLICES_STABILITY = 4

# Weging van stabiliteit in de objective:
# score = roi + STABILITY_WEIGHT * min_slice_roi
STABILITY_WEIGHT = 0.5

# Output-bestanden
OUTPUT_DIR = Path("output") / "research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "eci_autotune_gridsearch.csv"
BEST_JSON = OUTPUT_DIR / "candidate_rule_params.json"

# Optioneel: probeer eci_picks.py bij te werken
ECI_PICKS_PATH = Path("eci_picks.py")  # relative to current working dir

# Alleen HOME/AWAY of ook DRAW meenemen
INCLUDE_DRAW = False

# Drift-support settings (zelfde idee als eci_picks.py)
DRIFT_SUPPORT_THRESHOLD = -0.03
DRIFT_OPPOSE_THRESHOLD  =  0.03

DRIFT_SUPPORT_BONUS     = 0.10
DRIFT_OPPOSE_PENALTY    = 0.10

SNAP_BONUS_THRESHOLD    = 15
SNAP_BONUS              = 0.05

RANGE_PENALTY_THRESHOLD = 0.50
RANGE_PENALTY           = 0.05

# We tunen nu ook op eind-strength
MIN_STRENGTH_LIST = [1.5, 2.0, 2.5]

# ===================== Parameter-grid =====================

@dataclass
class ParamGrid:
    min_probs: List[float]
    min_values: List[float]
    min_rating_gaps: List[int]
    min_odds: List[float]
    max_odds: List[float]
    min_snapshots_list: List[int]
    min_drift_abs_list: List[float]
    min_strength_list: List[float]

PARAM_GRID = ParamGrid(
    min_probs=[0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58],
    min_values=[1.04, 1.08, 1.12, 1.16],
    min_rating_gaps=[0, 200, 400],
    min_odds=[1.40, 1.60, 1.80],
    max_odds=[4.0, 5.0, 6.0],
    min_snapshots_list=[4, 7, 10],
    min_drift_abs_list=[0.00, 0.03, 0.06],
    min_strength_list=MIN_STRENGTH_LIST,
)

# ==========================================================
#   Data ophalen & voorbereiden
# ==========================================================

QUERY = """
SELECT
    match_id,
    date,
    competition,
    home_team,
    away_team,
    odds_home,
    odds_draw,
    odds_away,
    home_win_pct,
    draw_pct,
    away_win_pct,
    score,
    home_rating,
    away_rating,
    home_drift_pct,
    away_drift_pct,
    home_drift_abs,
    away_drift_abs,
    home_range,
    away_range,
    n_snapshots,
    hours_stale,
    market_age_hours,
    home_last_move_pct,
    away_last_move_pct,
    home_recent24_pct,
    away_recent24_pct,
    kickoff_at,
    scrape_to_kickoff_hours
FROM public.betmobile_api_ready_mv
WHERE odds_home IS NOT NULL
  AND odds_draw IS NOT NULL
  AND odds_away IS NOT NULL
  AND home_win_pct IS NOT NULL
  AND draw_pct IS NOT NULL
  AND away_win_pct IS NOT NULL
  AND score IS NOT NULL
"""


def parse_score_to_code(score: Optional[str]) -> Optional[str]:
    if not score or not isinstance(score, str):
        return None
    m = re.findall(r"(\d+)", score)
    if len(m) >= 2:
        h, a = int(m[0]), int(m[1])
        if h > a:
            return "H"
        if h < a:
            return "A"
        return "D"
    return None


def normalize_result(raw: Optional[str], fallback_score: Optional[str]) -> Optional[str]:
    if isinstance(raw, str):
        s = raw.strip().upper()
        mapping = {
            "HOME_WIN": "H",
            "AWAY_WIN": "A",
            "DRAW": "D",
            "H": "H",
            "HOME": "H",
            "1": "H",
            "A": "A",
            "AWAY": "A",
            "2": "A",
            "D": "D",
            "X": "D",
            "HOME WIN": "H",
            "AWAY WIN": "A",
            "DRAWN": "D",
            "DRAW GAME": "D",
        }
        if s in mapping:
            return mapping[s]
        if "HOME" in s and "WIN" in s:
            return "H"
        if "AWAY" in s and "WIN" in s:
            return "A"
        if "DRAW" in s:
            return "D"
    return parse_score_to_code(fallback_score)


def fetch_data() -> pd.DataFrame:
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    df = pd.read_sql(QUERY, engine)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # kansen mogelijk in 0–100
    for c in ["home_win_pct", "draw_pct", "away_win_pct"]:
        if df[c].dropna().gt(1.0).mean() > 0.5:
            df[c] = df[c] / 100.0

    for c in [
        "home_rating", "away_rating",
        "home_drift_pct", "away_drift_pct",
        "home_drift_abs", "away_drift_abs",
        "home_range", "away_range",
        "hours_stale"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["n_snapshots"] = pd.to_numeric(df["n_snapshots"], errors="coerce").fillna(0).astype(int)

    # sanity odds
    df = df[
        (df["odds_home"] > 1.01)
        & (df["odds_draw"] > 1.01)
        & (df["odds_away"] > 1.01)
    ].copy()

    # resultaat normaliseren
    # De huidige tuning-view bevat score, maar niet altijd result.
    # normalize_result gebruikt result als die bestaat, anders score als fallback.
    if "result" not in df.columns:
        df["result"] = None

    df["result_code"] = np.vectorize(normalize_result)(df["result"], df["score"])
    df = df[df["result_code"].isin(["H", "D", "A"])].copy()

    print("Rows after basic cleaning:", len(df))
    print("Result_code distribution:", df["result_code"].value_counts().to_dict())
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maak long-form:
    - outcome: 'home' / 'draw' / 'away'
    - true_code: 'H' / 'D' / 'A'
    - prob, odds, value
    - drift_pct: oddsdrift van de kant waarop je inzet
    - range_value: volatiliteit van de kant waarop je inzet
    """
    rating_diff_abs = (df["home_rating"] - df["away_rating"]).abs()
    rating_home_edge = df["home_rating"] - df["away_rating"]

    common = {
        "match_id": df["match_id"],
        "date": df["date"],
        "competition": df["competition"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_rating": df["home_rating"],
        "away_rating": df["away_rating"],
        "rating_diff_abs": rating_diff_abs,
        "rating_home_edge": rating_home_edge,
        "score": df["score"],
        "result_code": df["result_code"],
        "n_snapshots": df["n_snapshots"],
        "hours_stale": df["hours_stale"],
    }

    home = pd.DataFrame(
        {
            **common,
            "outcome": "home",
            "true_code": "H",
            "prob": df["home_win_pct"].astype(float),
            "odds": df["odds_home"].astype(float),
            "drift_pct": df["home_drift_pct"].astype(float),
            "drift_abs": df["home_drift_abs"].astype(float),
            "range_value": df["home_range"].astype(float),
            "edge_ok": (rating_home_edge >= 0),
        }
    )

    away = pd.DataFrame(
        {
            **common,
            "outcome": "away",
            "true_code": "A",
            "prob": df["away_win_pct"].astype(float),
            "odds": df["odds_away"].astype(float),
            "drift_pct": df["away_drift_pct"].astype(float),
            "drift_abs": df["away_drift_abs"].astype(float),
            "range_value": df["away_range"].astype(float),
            "edge_ok": True,
        }
    )

    frames = [home, away]

    if INCLUDE_DRAW:
        draw = pd.DataFrame(
            {
                **common,
                "outcome": "draw",
                "true_code": "D",
                "prob": df["draw_pct"].astype(float),
                "odds": df["odds_draw"].astype(float),
                "drift_pct": df[["home_drift_pct", "away_drift_pct"]].abs().max(axis=1).astype(float),
                "drift_abs": df[["home_drift_abs", "away_drift_abs"]].abs().max(axis=1).astype(float),
                "range_value": df[["home_range", "away_range"]].max(axis=1).astype(float),
                "edge_ok": True,
            }
        )
        frames.append(draw)

    long_df = pd.concat(frames, ignore_index=True)

    long_df["value"] = long_df["prob"] * long_df["odds"]
    long_df["won"] = long_df["result_code"] == long_df["true_code"]
    long_df["profit"] = np.where(long_df["won"], long_df["odds"] - 1.0, -1.0)

    return long_df


# ==========================================================
#   Bets selecteren op basis van parameters
# ==========================================================

def select_bets(
    long_df: pd.DataFrame,
    min_prob: float,
    min_value: float,
    min_rating_gap: int,
    min_odds: float,
    max_odds: float,
    min_snapshots: int,
    min_drift_abs: float,
    min_strength: float,
) -> pd.DataFrame:
    df = long_df.copy()

    # Basisregels
    mask = (
        (df["prob"] >= min_prob)
        & (df["value"] >= min_value)
        & (df["rating_diff_abs"] >= min_rating_gap)
        & (df["odds"] >= min_odds)
        & (df["odds"] <= max_odds)
        & (df["n_snapshots"] >= min_snapshots)
        & (df["drift_abs"] >= min_drift_abs)
        & (df["edge_ok"])
    )

    df = df.loc[mask].copy()
    if df.empty:
        return df

    # Simpele strength, in lijn met eci_picks.py
    df["raw_strength"] = (
        (df["value"] - min_value).clip(lower=0) * 2
        + (df["prob"] - min_prob).clip(lower=0) * 2
        + df["rating_diff_abs"].clip(lower=0) / 500
    )

    # Extra home-edge bonus alleen voor HOME
    df["raw_strength"] += np.where(
        df["outcome"] == "home",
        df["rating_home_edge"].clip(lower=0) / 500,
        0.0
    )

    df["strength_adj"] = df["raw_strength"]

    # Drift richting meenemen
    support_mask = df["drift_pct"] <= DRIFT_SUPPORT_THRESHOLD
    oppose_mask = df["drift_pct"] >= DRIFT_OPPOSE_THRESHOLD

    df["strength_adj"] += np.where(support_mask, DRIFT_SUPPORT_BONUS, 0.0)
    df["strength_adj"] -= np.where(oppose_mask, DRIFT_OPPOSE_PENALTY, 0.0)

    # Meer snapshots = klein plusje
    df["strength_adj"] += np.where(df["n_snapshots"] >= SNAP_BONUS_THRESHOLD, SNAP_BONUS, 0.0)

    # Grote range = klein minnetje
    df["strength_adj"] -= np.where(df["range_value"] >= RANGE_PENALTY_THRESHOLD, RANGE_PENALTY, 0.0)

    # Eindfilter
    df = df[df["strength_adj"] >= min_strength].copy()

    cols = [
        "match_id",
        "date",
        "competition",
        "home_team",
        "away_team",
        "outcome",
        "prob",
        "odds",
        "value",
        "result_code",
        "profit",
        "home_rating",
        "away_rating",
        "rating_diff_abs",
        "rating_home_edge",
        "score",
        "drift_pct",
        "drift_abs",
        "range_value",
        "n_snapshots",
        "hours_stale",
        "raw_strength",
        "strength_adj",
    ]

    return df.loc[:, cols].copy()


# ==========================================================
#   Evaluatie & stabiliteit
# ==========================================================

def compute_basic_stats(bets: pd.DataFrame) -> Dict[str, float]:
    n = len(bets)
    if n == 0:
        return {"bets": 0, "roi": 0.0, "hitrate": 0.0, "profit": 0.0}

    profit = bets["profit"].sum()
    roi = profit / n
    hitrate = (bets["profit"] > 0).mean()

    return {"bets": n, "roi": roi, "hitrate": hitrate, "profit": profit}


def compute_stability(bets: pd.DataFrame, n_slices: int = N_SLICES_STABILITY) -> float:
    """
    Splits bets in n_slices op tijd en neemt de minimale ROI over die slices.
    """
    if bets.empty or n_slices < 2:
        return 0.0

    bets_sorted = bets.sort_values("date").reset_index(drop=True)
    idx = np.linspace(0, len(bets_sorted), n_slices + 1, dtype=int)
    slice_rois: List[float] = []

    for i in range(n_slices):
        start, end = idx[i], idx[i + 1]
        if end - start < 30:
            # te weinig volume in deze slice, sla over
            continue
        sl = bets_sorted.iloc[start:end]
        stats = compute_basic_stats(sl)
        slice_rois.append(stats["roi"])

    if not slice_rois:
        return 0.0

    return float(min(slice_rois))


def evaluate_combo(
    long_df: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    bets = select_bets(
        long_df=long_df,
        min_prob=params["min_prob"],
        min_value=params["min_value"],
        min_rating_gap=params["min_rating_gap"],
        min_odds=params["min_odds"],
        max_odds=params["max_odds"],
        min_snapshots=params["min_snapshots"],
        min_drift_abs=params["min_drift_abs"],
        min_strength=params["min_strength"],
    )

    stats = compute_basic_stats(bets)
    min_slice_roi = compute_stability(bets, N_SLICES_STABILITY)

    score = stats["roi"] + STABILITY_WEIGHT * min_slice_roi

    result = {
        **params,
        "bets": stats["bets"],
        "profit": stats["profit"],
        "roi": stats["roi"],
        "hitrate": stats["hitrate"],
        "min_slice_roi": min_slice_roi,
        "score": score,
    }
    return result


# ==========================================================
#   Gridsearch
# ==========================================================

def gridsearch(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grid = PARAM_GRID

    total = (
        len(grid.min_probs)
        * len(grid.min_values)
        * len(grid.min_rating_gaps)
        * len(grid.min_odds)
        * len(grid.max_odds)
        * len(grid.min_snapshots_list)
        * len(grid.min_drift_abs_list)
        * len(grid.min_strength_list)
    )
    done = 0

    for min_prob in grid.min_probs:
        for min_value in grid.min_values:
            for min_rating_gap in grid.min_rating_gaps:
                for min_odds in grid.min_odds:
                    for max_odds in grid.max_odds:
                        if max_odds <= min_odds + 0.01:
                            continue  # onzinnige combo
                        for min_snapshots in grid.min_snapshots_list:
                            for min_drift_abs in grid.min_drift_abs_list:
                                for min_strength in grid.min_strength_list:
                                    done += 1
                                    if done % 200 == 0 or done == total:
                                        print(f"[grid] {done}/{total} combinaties...")

                                    params = {
                                        "min_prob": float(min_prob),
                                        "min_value": float(min_value),
                                        "min_rating_gap": int(min_rating_gap),
                                        "min_odds": float(min_odds),
                                        "max_odds": float(max_odds),
                                        "min_snapshots": int(min_snapshots),
                                        "min_drift_abs": float(min_drift_abs),
                                        "min_strength": float(min_strength),
                                    }

                                    res = evaluate_combo(long_df, params)
                                    if res["bets"] < MIN_BETS_GLOBAL:
                                        continue

                                    rows.append(res)

    if not rows:
        print("⚠️ Geen enkele parametercombinatie haalde de MIN_BETS_GLOBAL drempel.")
        return pd.DataFrame()

    df_res = pd.DataFrame(rows)
    df_res.sort_values(
        by=["score", "roi", "min_slice_roi", "profit", "bets"],
        ascending=[False, False, False, False, False],
        inplace=True,
    )

    df_res.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Gridsearch-resultaten geschreven naar: {RESULTS_CSV}")
    return df_res


# ==========================================================
#   eci_picks.py update helper
# ==========================================================

def update_eci_picks(best_params: Dict[str, Any]) -> None:
    """
    Probeert in eci_picks.py een blok als:

        ECI_RULE_PARAMS = {
            "min_prob": ...,
            ...
        }

    bij te werken. Als dat niet lukt, wordt alleen JSON aangemaakt.
    """
    if not ECI_PICKS_PATH.exists():
        print(f"ℹ️ {ECI_PICKS_PATH} niet gevonden; sla alleen JSON op.")
        return

    text = ECI_PICKS_PATH.read_text(encoding="utf-8")

    pattern = r"(ECI_RULE_PARAMS\s*=\s*)(\{.*?\})"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        print("ℹ️ Geen ECI_RULE_PARAMS blok gevonden in eci_picks.py; sla alleen JSON op.")
        return

    prefix = m.group(1)
    dict_str = m.group(2)

    try:
        import ast

        existing = ast.literal_eval(dict_str)
        if not isinstance(existing, dict):
            raise ValueError("ECI_RULE_PARAMS is geen dict")

    except Exception as e:
        print(f"⚠️ Kon ECI_RULE_PARAMS niet parsen ({e}); sla alleen JSON op.")
        return

    # Update relevante keys
    mapping = {
        "min_prob": "min_prob",
        "min_value": "min_value",
        "min_rating_gap": "min_rating_gap",
        "min_odds": "min_odds",
        "max_odds": "max_odds",
        "min_snapshots": "min_snapshots",
        "min_drift_abs": "min_drift_abs",
    }
    for k_cfg, k_best in mapping.items():
        existing[k_cfg] = float(best_params[k_best]) if "value" in k_cfg or "prob" in k_cfg or "odds" in k_cfg else int(best_params[k_best])

    new_dict_str = json.dumps(existing, indent=4, sort_keys=True)
    new_block = prefix + new_dict_str

    new_text = text[: m.start()] + new_block + text[m.end() :]

    # Probeer ook MIN_STRENGTH bij te werken
    ms_pattern = r"(MIN_STRENGTH\s*=\s*)([0-9]+(?:\.[0-9]+)?)"
    ms_match = re.search(ms_pattern, new_text)
    if ms_match:
        new_text = re.sub(
            ms_pattern,
            rf"\g<1>{float(best_params['min_strength'])}",
            new_text,
            count=1
        )

    backup_path = ECI_PICKS_PATH.with_suffix(".py.bak")
    backup_path.write_text(text, encoding="utf-8")
    ECI_PICKS_PATH.write_text(new_text, encoding="utf-8")

    print(f"✅ eci_picks.py bijgewerkt (backup: {backup_path.name}) met:")
    print(json.dumps(existing, indent=4))
    if ms_match:
        print(f"✅ MIN_STRENGTH bijgewerkt naar {float(best_params['min_strength'])}")


# ==========================================================
#   Main
# ==========================================================

def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)

    print("=== ECI AutoTune: data ophalen ===")
    df = fetch_data()
    if df.empty:
        print("⚠️ Geen data opgehaald; stop.")
        return

    print("=== Naar long-form ===")
    long_df = to_long(df)
    print("Long-form rows:", len(long_df))

    print("=== Gridsearch starten ===")
    results = gridsearch(long_df)
    if results.empty:
        print("⚠️ Geen bruikbare resultaten uit gridsearch.")
        return

    best = results.iloc[0].to_dict()

    print("\n=== BESTE PARAMETERSET ===")
    for k in ["min_prob", "min_value", "min_rating_gap", "min_odds", "max_odds", "min_snapshots", "min_drift_abs", "min_strength"]:
        print(f"{k}: {best[k]}")

    print(
        f"\nBets: {int(best['bets'])}, ROI: {best['roi']:.4f}, "
        f"Min-slice ROI: {best['min_slice_roi']:.4f}, Score: {best['score']:.4f}, "
        f"Profit: {best['profit']:.2f}"
    )

    # JSON wegschrijven
    out_cfg = {
        "min_prob": float(best["min_prob"]),
        "min_value": float(best["min_value"]),
        "min_rating_gap": int(best["min_rating_gap"]),
        "min_odds": float(best["min_odds"]),
        "max_odds": float(best["max_odds"]),
        "min_snapshots": int(best["min_snapshots"]),
        "min_drift_abs": float(best["min_drift_abs"]),
        "min_strength": float(best["min_strength"]),
        "metrics": {
            "bets": int(best["bets"]),
            "profit": float(best["profit"]),
            "roi": float(best["roi"]),
            "min_slice_roi": float(best["min_slice_roi"]),
            "score": float(best["score"]),
            "hitrate": float(best["hitrate"]),
        },
    }
    Path(BEST_JSON).write_text(json.dumps(out_cfg, indent=4), encoding="utf-8")
    print(f"\n✅ Beste parameters + metrics geschreven naar: {BEST_JSON}")

    # Niet automatisch productiecode aanpassen.
    # Vanaf nu schrijft autotune alleen candidate output weg.
    print("ℹ️ Productiecode niet automatisch bijgewerkt. Gebruik later promote/release-flow.")

if __name__ == "__main__":
    main()
