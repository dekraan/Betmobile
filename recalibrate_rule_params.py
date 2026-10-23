"""
recalibrate_rule_params.py

Herijking van de ECI-regeldrempels (ECI_RULE_PARAMS) op de VOLLEDIGE
wedstrijdpopulatie uit betmobile_api_ready_mv, niet alleen op picks.

Wat het doet
------------
1. Laadt alle gespeelde wedstrijden vóór de cutoff (standaard 2026-08-18,
   het begin van de freeze). Data van daarna wordt geweigerd tenzij je
   expliciet --allow-freeze-data meegeeft.
2. Maakt per wedstrijd twee kandidaat-bets: HOME en AWAY.
3. Past de huidige regels toe (zelfde logica als rules.apply_rules).
4. Varieert per parameter één drempel tegelijk; de rest blijft op de
   huidige waarde staan.
5. Meet per variant:
     resid   = gemiddeld (gewonnen - Shin-gedevigde Bet365-kans)
               > 0: de selectie wint vaker dan de markt verwacht
     roi     = gemiddelde winst per inzet bij Bet365-odds
   met cluster-robuuste SE per speeldag en t(G-1)-kritieke waarden.
6. Een nieuwe drempel wordt alleen geadviseerd als die zowel genoeg bets
   heeft als de huidige drempel met marge verslaat. Anders: "houden".

Wat het NIET doet
-----------------
- Niets wegschrijven naar de database en niets aan config.py veranderen.
  Output: console + CSV + JSON-voorstel in output/research/.
- De ±30 drift/strength-knoppen en min_snapshots worden niet herijkt.
- Geen gezamenlijke optimalisatie: één parameter tegelijk, bewust,
  om ruis-fitten te beperken.

Gebruik
-------
    python recalibrate_rule_params.py
    python recalibrate_rule_params.py --cutoff 2026-08-18 --min-bets 100
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import stats

from sqlalchemy import create_engine

# Werkt zowel vanuit de projectroot als vanuit eci_engine/.
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "eci_engine", _HERE, _HERE.parent):
    if (_p / "config.py").exists() and (_p / "rules.py").exists():
        sys.path.insert(0, str(_p))
        break

from config import ECI_RULE_PARAMS, DRIFT_OPPOSE_THRESHOLD, OUTPUT_DIR, DB_DSN
from db import get_upcoming_source_name
from prob_calibration import devig_shin


def db_engine():
    """Eigen engine, read-only: dit script kan niets wijzigen in de database."""
    return create_engine(
        DB_DSN,
        connect_args={"options": "-c default_transaction_read_only=on"},
    )

FREEZE_START = date(2026, 8, 18)
EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Kandidaatwaarden per parameter. Bewust grof: weinig stappen = minder ruis.
GRIDS = {
    "min_prob":       [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.70],
    "min_value":      [0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10],
    "min_rating_gap": [0, 100, 200, 300, 400, 500, 600, 700, 800],
    "min_odds":       [1.30, 1.40, 1.50, 1.60, 1.70, 1.80],
    "max_odds":       [2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00],
}


# =====================================================================
# LADEN EN VOORBEREIDEN
# =====================================================================

def load_population(cutoff: date) -> pd.DataFrame:
    source = get_upcoming_source_name()
    q = f"""
        SELECT match_id, date, competition,
               odds_home, odds_draw, odds_away,
               home_win_pct, draw_pct, away_win_pct,
               home_rating, away_rating, score,
               home_drift_pct, away_drift_pct, n_snapshots
        FROM public.{source}
        WHERE score IS NOT NULL
          AND odds_home IS NOT NULL AND odds_draw IS NOT NULL AND odds_away IS NOT NULL
          AND to_date(date, 'YYYY-MM-DD') < DATE '{cutoff.isoformat()}'
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    print(f"[load] bron = {source}, {len(df)} gespeelde wedstrijden vóór {cutoff}")
    return df


def parse_result(score) -> str | None:
    m = pd.Series(str(score)).str.extract(r"(\d+)\D+(\d+)").iloc[0]
    if m.isna().any():
        return None
    h, a = int(m.iloc[0]), int(m.iloc[1])
    return "H" if h > a else ("A" if a > h else "D")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["home_win_pct", "draw_pct", "away_win_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].max() > 1.2:
            df[c] = df[c] / 100.0

    for c in ["odds_home", "odds_draw", "odds_away", "home_drift_pct",
              "away_drift_pct", "n_snapshots"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["home_rating"] = pd.to_numeric(df["home_rating"], errors="coerce")
    df["away_rating"] = pd.to_numeric(df["away_rating"], errors="coerce")
    df["rating_gap"] = (df["home_rating"] - df["away_rating"]).abs()
    df["rating_home_edge"] = df["home_rating"] - df["away_rating"]

    df["result"] = df["score"].map(parse_result)

    odds = df[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    ok = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    mkt = np.full_like(odds, np.nan)
    if ok.any():
        mkt[ok], _ = devig_shin(odds[ok])
    df["mkt_home"], df["mkt_draw"], df["mkt_away"] = mkt[:, 0], mkt[:, 1], mkt[:, 2]

    n0 = len(df)
    df = df.dropna(subset=["result", "home_win_pct", "away_win_pct", "rating_gap", "mkt_home"])
    if len(df) < n0:
        print(f"[prepare] {n0 - len(df)} wedstrijden afgevallen (geen uitslag/kansen/odds/rating)")
    return df


def to_bets(df: pd.DataFrame) -> pd.DataFrame:
    """Eén rij per kandidaat-bet: HOME en AWAY."""
    common = ["match_id", "date", "competition", "rating_gap", "n_snapshots"]

    home = df[common].copy()
    home["side"] = "HOME"
    home["p_eci"] = df["home_win_pct"]
    home["odds"] = df["odds_home"]
    home["p_mkt"] = df["mkt_home"]
    home["drift"] = df["home_drift_pct"]
    home["edge_ok"] = df["rating_home_edge"] >= 0
    home["won"] = (df["result"] == "H").astype(int)

    away = df[common].copy()
    away["side"] = "AWAY"
    away["p_eci"] = df["away_win_pct"]
    away["odds"] = df["odds_away"]
    away["p_mkt"] = df["mkt_away"]
    away["drift"] = df["away_drift_pct"]
    away["edge_ok"] = True          # rules.py heeft voor AWAY geen edge-check
    away["won"] = (df["result"] == "A").astype(int)

    bets = pd.concat([home, away], ignore_index=True)
    bets["value"] = bets["odds"] * bets["p_eci"]
    return bets


# =====================================================================
# REGELS EN METRIEKEN
# =====================================================================

def select(bets: pd.DataFrame, p: dict) -> pd.Series:
    """Zelfde voorwaarden als rules.apply_rules (zonder strength-score)."""
    return (
        (bets["p_eci"] >= p["min_prob"])
        & (bets["value"] >= p["min_value"])
        & (bets["rating_gap"] >= p["min_rating_gap"])
        & bets["odds"].between(p["min_odds"], p["max_odds"])
        & (bets["n_snapshots"].fillna(0) >= p["min_snapshots"])
        & (bets["drift"].abs().fillna(0) >= p["min_drift_abs"])
        & (bets["drift"].fillna(0) <= DRIFT_OPPOSE_THRESHOLD)
        & bets["edge_ok"]
    )


def cluster_mean(x: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """Gemiddelde met cluster-robuuste SE. Geeft (mean, se, G)."""
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    m = x.mean()
    d = pd.Series(x - m).groupby(groups).sum().to_numpy()
    G = len(d)
    if G < 2:
        return m, np.nan, G
    se = np.sqrt((d ** 2).sum()) / n * np.sqrt(G / (G - 1))
    return m, se, G


def evaluate(bets: pd.DataFrame, mask: pd.Series) -> dict:
    s = bets[mask]
    n = len(s)
    if n == 0:
        return {"n": 0, "resid": np.nan, "resid_se": np.nan, "resid_t": np.nan,
                "roi": np.nan, "roi_se": np.nan, "G": 0, "t_crit": np.nan}
    g = s["date"].to_numpy()
    resid, resid_se, G = cluster_mean((s["won"] - s["p_mkt"]).to_numpy(float), g)
    profit = np.where(s["won"] == 1, s["odds"] - 1.0, -1.0)
    roi, roi_se, _ = cluster_mean(profit.astype(float), g)
    t_crit = stats.t.ppf(0.975, G - 1) if G >= 2 else np.nan
    return {
        "n": n,
        "G": G,
        "resid": resid,
        "resid_se": resid_se,
        "resid_t": resid / resid_se if resid_se and resid_se > 0 else np.nan,
        "t_crit": t_crit,
        "roi": roi,
        "roi_se": roi_se,
    }


# =====================================================================
# HOOFDLUS
# =====================================================================

def run(cutoff: date, min_bets: int) -> tuple[pd.DataFrame, dict]:
    df = prepare(load_population(cutoff))
    bets = to_bets(df)
    print(f"[bets] {len(bets)} kandidaat-bets uit {df['match_id'].nunique()} wedstrijden")

    current = dict(ECI_RULE_PARAMS)
    base = evaluate(bets, select(bets, current))
    print(
        f"\n[huidig] n={base['n']}  resid={base['resid']:+.4f} (se {base['resid_se']:.4f})  "
        f"roi={base['roi']:+.3f}"
    )

    rows = []
    proposal = {}
    for param, grid in GRIDS.items():
        for val in grid:
            p = dict(current)
            p[param] = val
            if p["min_odds"] >= p["max_odds"]:
                continue
            r = evaluate(bets, select(bets, p))
            r.update({"param": param, "value": val, "is_current": val == current[param]})
            rows.append(r)

        sub = pd.DataFrame([r for r in rows if r["param"] == param])
        eligible = sub[(sub["n"] >= min_bets) & sub["resid"].notna()]
        if eligible.empty:
            proposal[param] = {"advies": "houden", "waarde": current[param],
                               "reden": f"geen variant met >= {min_bets} bets"}
            continue

        best = eligible.loc[eligible["resid"].idxmax()]
        cur_row = sub[sub["is_current"]]
        cur_resid = float(cur_row["resid"].iloc[0]) if not cur_row.empty else base["resid"]
        verbetering = float(best["resid"]) - cur_resid
        drempel = float(best["resid_se"])   # verbetering moet groter zijn dan 1 SE

        if best["value"] != current[param] and verbetering > drempel:
            proposal[param] = {
                "advies": "wijzigen",
                "waarde": float(best["value"]),
                "huidig": current[param],
                "verbetering_resid": round(verbetering, 4),
                "se": round(drempel, 4),
                "n": int(best["n"]),
            }
        else:
            proposal[param] = {
                "advies": "houden",
                "waarde": current[param],
                "beste_alternatief": float(best["value"]),
                "verbetering_resid": round(verbetering, 4),
                "se": round(drempel, 4),
                "reden": "verschil kleiner dan 1 SE" if best["value"] != current[param]
                         else "huidige waarde is al de beste",
            }

    return pd.DataFrame(rows), {"huidig": base, "voorstel": proposal}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=FREEZE_START.isoformat(),
                    help="Alleen wedstrijden vóór deze datum (YYYY-MM-DD)")
    ap.add_argument("--min-bets", type=int, default=100,
                    help="Minimum aantal bets voor een variant om mee te tellen")
    ap.add_argument("--allow-freeze-data", action="store_true",
                    help="Sta een cutoff na het begin van de freeze toe")
    args = ap.parse_args()

    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d").date()
    if cutoff > FREEZE_START and not args.allow_freeze_data:
        raise SystemExit(
            f"Cutoff {cutoff} valt na het begin van de freeze ({FREEZE_START}). "
            "Dat zou meetdata gebruiken. Geef --allow-freeze-data mee als dat echt de bedoeling is."
        )

    table, result = run(cutoff, args.min_bets)

    pd.set_option("display.width", 160)
    for param in GRIDS:
        sub = table[table["param"] == param][
            ["value", "n", "resid", "resid_se", "resid_t", "roi", "is_current"]
        ]
        print(f"\n=== {param} ===")
        print(sub.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print("\n=== VOORSTEL ===")
    for param, v in result["voorstel"].items():
        print(f"  {param:16s} {v['advies']:9s} -> {v['waarde']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = EXPORT_DIR / f"recalibration_{stamp}.csv"
    json_path = EXPORT_DIR / f"recalibration_{stamp}.json"
    table.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cutoff": cutoff.isoformat(),
                "min_bets": args.min_bets,
                "huidige_params": ECI_RULE_PARAMS,
                "huidig_resultaat": result["huidig"],
                "voorstel": result["voorstel"],
            },
            f, indent=2, default=float, ensure_ascii=False,
        )
    print(f"\n[export] {csv_path}\n[export] {json_path}")


if __name__ == "__main__":
    main()