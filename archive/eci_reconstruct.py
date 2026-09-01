"""
eci_reconstruct.py
==================

Reconstrueert ECI-wedstrijdkansen uit de ratinghistorie, terug tot 2007.

Werkwijze
---------
1. Parameters (H, sigma, tau) worden per competitie gefit op eci_data,
   waar zowel de ratings als de gepubliceerde kansen in staan.
2. Voor elke wedstrijd wordt de rating opgezocht in ratings_history.jsonl:
   het laatste weekpunt STRIKT VOOR de wedstrijddatum. Dat vermijdt
   lookahead, want het weekpunt van de wedstrijddag zelf verwerkt de
   uitslag al.
3. Met die ratings en de parameters volgt de kans.

Wat je hiervoor nodig hebt
--------------------------
Een wedstrijdlijst. De ratings gaan terug tot juli 2007, maar wedstrijden
ken je alleen vanaf februari 2025 uit eci_data. Voor oudere seizoenen heb
je een externe bron nodig; football-data.co.uk levert fixtures, uitslagen
en slotodds voor ruim twintig competities vanaf de jaren negentig.

Gebruik
-------
    python eci_reconstruct.py coverage
        Hoeveel teams hebben in welk jaar een rating.

    python eci_reconstruct.py selftest
        Reconstrueer de kansen van eci_data volledig uit ratings_history
        en vergelijk met de opgeslagen kansen. Dit toetst de hele keten:
        naamkoppeling, ratingopzoeking en model.

    python eci_reconstruct.py fixtures --csv wedstrijden.csv
        Kolommen: date, home_team, away_team, competition
        Schrijft reconstructed_probs.csv weg.

    python eci_reconstruct.py params
        Toon de gefitte parameters per competitie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
    from scipy.optimize import minimize
except ImportError:
    sys.exit("scipy ontbreekt: pip install scipy")

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 ontbreekt: pip install psycopg2-binary")

try:
    from betmobile_settings import DB_CONFIG
except ImportError:
    sys.exit("betmobile_settings.py niet gevonden. Draai vanuit je Betmobile-map.")


HERE = Path(__file__).resolve().parent
RATINGS = HERE / "eci_history" / "ratings_history.jsonl"

# Globale parameters, gefit op de match-odds feed. Terugval als een
# competitie te weinig wedstrijden heeft voor een eigen fit.
GLOBAL = (182.9, 1026.2, 0.3467)
MIN_MATCHES = 150


def hr(t): print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def norm_name(s) -> str:
    """Namen vergelijkbaar maken. Beide bronnen komen van ECI, dus
    afwijkingen zijn zeldzaam, maar accenten en spaties variëren."""
    import unicodedata
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(s.lower().split())


def probs(H, sigma, tau, gap):
    z = (np.asarray(gap, dtype=float) + H) / sigma
    p_away = norm.cdf(-tau - z)
    p_home = 1.0 - norm.cdf(tau - z)
    return p_home, 1.0 - p_home - p_away, p_away


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_ratings():
    if not RATINGS.exists():
        sys.exit(f"Niet gevonden: {RATINGS}\nDraai eerst eci_history_collector.py collect")

    print(f"  ratings inlezen uit {RATINGS.name} ...")
    df = pd.read_json(RATINGS, lines=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["eci_team_id", "date"])

    series = {}
    for tid, sub in df.groupby("eci_team_id"):
        series[str(tid)] = (sub["date"].values, sub["points"].values)

    # Naam -> ID. Bij dubbele namen (er zijn twee clubs "Arsenal") houden
    # we de eerste; de zelftest rapporteert hoeveel er misgaat.
    names = {}
    for tid, name in df[["eci_team_id", "team_name"]].drop_duplicates().values:
        key = norm_name(name)
        if key:
            names.setdefault(key, str(tid))

    print(f"  {len(df):,} ratingpunten, {len(series):,} teams, "
          f"{df.date.min():%Y-%m-%d} tot {df.date.max():%Y-%m-%d}")
    return series, names


def rating_before(series, tid: str, when) -> float | None:
    """Laatste weekpunt STRIKT VOOR de wedstrijddatum."""
    entry = series.get(str(tid))
    if entry is None:
        return None
    dates, pts = entry
    i = np.searchsorted(dates, np.datetime64(when), side="left")
    return float(pts[i - 1]) if i > 0 else None


def load_eci_data() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(readonly=True, autocommit=True)
    df = pd.read_sql("""
        SELECT date, home_team, away_team, competition,
               home_rating, away_rating,
               home_win_pct, draw_pct, away_win_pct
        FROM eci_data
        WHERE home_rating IS NOT NULL AND away_rating IS NOT NULL
          AND home_win_pct IS NOT NULL
          AND COALESCE(is_excluded, false) = false
    """, conn)
    conn.close()

    for c in ("home_rating", "away_rating"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_rating", "away_rating", "date"])
    if df["home_win_pct"].max() > 1.5:
        for c in ("home_win_pct", "draw_pct", "away_win_pct"):
            df[c] = df[c] / 100.0
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def fit_one(sub, x0=GLOBAL):
    gap = (sub.home_rating - sub.away_rating).values
    pH, pD, pA = (sub.home_win_pct.values, sub.draw_pct.values,
                  sub.away_win_pct.values)

    def loss(p):
        if p[1] <= 1 or p[2] <= 0:
            return 1e9
        ph, pd_, pa = probs(p[0], p[1], p[2], gap)
        return np.mean((ph - pH) ** 2 + (pd_ - pD) ** 2 + (pa - pA) ** 2)

    r = minimize(loss, x0, method="Nelder-Mead",
                 options={"maxiter": 20000, "fatol": 1e-14, "xatol": 1e-9})
    return tuple(r.x)


def fit_params(df) -> dict:
    params = {}
    for comp, sub in df.groupby("competition"):
        if len(sub) >= MIN_MATCHES:
            params[comp] = fit_one(sub)
    return params


def params_for(params: dict, comp) -> tuple:
    return params.get(comp, GLOBAL)


# ---------------------------------------------------------------------------
# Commando's
# ---------------------------------------------------------------------------

def cmd_coverage():
    hr("Dekking van de ratinghistorie")
    series, _ = load_ratings()

    rows = []
    for year in range(2007, 2027):
        stamp = np.datetime64(f"{year}-08-15")
        n = sum(1 for tid in series
                if rating_before(series, tid, stamp) is not None)
        rows.append((year, n))

    print(f"\n  {'jaar':<6} {'teams met rating':>18}")
    for year, n in rows:
        bar = "#" * int(n / 25)
        print(f"  {year:<6} {n:>18,}  {bar}")

    print("\n  Dit is wat je kunt reconstrueren, mits je fixtures hebt.")
    print("  Zonder wedstrijdlijst zijn ratings alleen getallen.")


def cmd_params():
    hr("Parameters per competitie")
    df = load_eci_data()
    params = fit_params(df)
    print(f"  gefit op {len(df):,} wedstrijden, {len(params)} competities\n")
    print(f"  {'competitie':<22} {'n':>6} {'H':>8} {'sigma':>9} {'tau':>8} {'H/sigma':>8}")
    for comp in sorted(params, key=lambda c: -(params[c][0] / params[c][1])):
        H, s, tau = params[comp]
        n = (df.competition == comp).sum()
        print(f"  {str(comp)[:22]:<22} {n:>6,} {H:>8.1f} {s:>9.1f} "
              f"{tau:>8.4f} {H/s:>8.3f}")
    print(f"\n  terugval voor de rest: H={GLOBAL[0]} sigma={GLOBAL[1]} tau={GLOBAL[2]}")


def cmd_selftest():
    hr("Zelftest: volledige keten")
    print("  Kansen worden opnieuw berekend uit ratings_history, dus zonder")
    print("  de ratings die al in eci_data staan. Dat toetst ook de")
    print("  naamkoppeling en de ratingopzoeking.\n")

    series, names = load_ratings()
    df = load_eci_data()
    params = fit_params(df)
    print(f"  {len(df):,} wedstrijden, {len(params)} competities met eigen parameters")

    hit_h = df.home_team.map(lambda n: names.get(norm_name(n)))
    hit_a = df.away_team.map(lambda n: names.get(norm_name(n)))
    linked = hit_h.notna() & hit_a.notna()
    print(f"  teams gekoppeld op naam: {linked.sum():,} / {len(df):,} "
          f"({linked.mean()*100:.1f}%)")

    if linked.sum() == 0:
        print("  Geen koppelingen; controleer de teamnamen.")
        return

    sub = df[linked].copy()
    sub["hid"], sub["aid"] = hit_h[linked], hit_a[linked]

    rh = [rating_before(series, r.hid, r.date) for r in sub.itertuples()]
    ra = [rating_before(series, r.aid, r.date) for r in sub.itertuples()]
    sub["rh"], sub["ra"] = rh, ra
    sub = sub.dropna(subset=["rh", "ra"])
    print(f"  met rating voor de wedstrijddatum: {len(sub):,}")

    # Hoe goed komt de opgezochte rating overeen met die in eci_data?
    d_h = (sub.rh - sub.home_rating).abs()
    print(f"\n  |rating uit historie - rating in eci_data|")
    print(f"    mediaan {d_h.median():6.2f}   gemiddeld {d_h.mean():6.2f}   "
          f"p95 {d_h.quantile(.95):6.2f}")

    ph = np.empty(len(sub))
    for comp, idx in sub.groupby("competition").groups.items():
        H, s, tau = params_for(params, comp)
        rows = sub.loc[idx]
        ph[sub.index.get_indexer(idx)] = probs(H, s, tau, rows.rh - rows.ra)[0]

    err = ph - sub.home_win_pct.values
    print(f"\n  gereconstrueerde kans versus gepubliceerde kans")
    print(f"    MAE      {np.abs(err).mean()*100:5.2f} pp")
    print(f"    mediaan  {np.median(np.abs(err))*100:5.2f} pp")
    print(f"    p95      {np.percentile(np.abs(err),95)*100:5.2f} pp")
    print(f"    R2       {1 - np.var(err)/np.var(sub.home_win_pct.values):.4f}")

    out = sub.assign(recon_home=ph, fout=err)
    per = (out.groupby("competition")
              .agg(n=("fout", "size"), mae=("fout", lambda s: s.abs().mean()*100))
              .sort_values("mae"))
    print(f"\n  beste en slechtste competities:")
    for comp, r in list(per.head(5).iterrows()) + list(per.tail(5).iterrows()):
        print(f"    {str(comp)[:24]:<24} {int(r.n):>6,}  {r.mae:5.2f} pp")

    path = HERE / "reconstruction_selftest.csv"
    out[["date", "competition", "home_team", "away_team", "rh", "ra",
         "home_win_pct", "recon_home", "fout"]].to_csv(path, index=False)
    print(f"\n  [OK] details -> {path.name}")


def cmd_fixtures(csv_path: str):
    hr("Reconstructie voor een wedstrijdlijst")
    fx = pd.read_csv(csv_path)
    need = {"date", "home_team", "away_team"}
    if not need.issubset(fx.columns):
        sys.exit(f"CSV mist kolommen. Nodig: {sorted(need)} (+ optioneel competition)")
    fx["date"] = pd.to_datetime(fx["date"])
    if "competition" not in fx:
        fx["competition"] = None
    print(f"  {len(fx):,} wedstrijden, {fx.date.min():%Y-%m-%d} tot {fx.date.max():%Y-%m-%d}")

    series, names = load_ratings()
    params = fit_params(load_eci_data())

    fx["hid"] = fx.home_team.map(lambda n: names.get(norm_name(n)))
    fx["aid"] = fx.away_team.map(lambda n: names.get(norm_name(n)))
    miss = fx.hid.isna() | fx.aid.isna()
    if miss.any():
        print(f"  [--] {miss.sum():,} niet gekoppeld op naam, bv:")
        for _, r in fx[miss].head(5).iterrows():
            print(f"       {r.home_team} - {r.away_team}")

    fx["rh"] = [rating_before(series, h, d) if pd.notna(h) else None
                for h, d in zip(fx.hid, fx.date)]
    fx["ra"] = [rating_before(series, a, d) if pd.notna(a) else None
                for a, d in zip(fx.aid, fx.date)]

    ok = fx.rh.notna() & fx.ra.notna()
    print(f"  bruikbaar: {ok.sum():,} / {len(fx):,} ({ok.mean()*100:.1f}%)")

    fx["p_home"] = np.nan; fx["p_draw"] = np.nan; fx["p_away"] = np.nan
    for comp, idx in fx[ok].groupby("competition", dropna=False).groups.items():
        H, s, tau = params_for(params, comp)
        rows = fx.loc[idx]
        ph, pd_, pa = probs(H, s, tau, rows.rh - rows.ra)
        fx.loc[idx, ["p_home", "p_draw", "p_away"]] = np.column_stack([ph, pd_, pa])

    path = HERE / "reconstructed_probs.csv"
    fx.to_csv(path, index=False)
    print(f"\n  [OK] -> {path.name}")
    print("  Kansen zijn zonder marge; voor eerlijke odds is 1/p de faire quote.")


def main():
    p = argparse.ArgumentParser(description="Reconstrueer ECI-kansen uit ratinghistorie")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("coverage", help="dekking van de ratinghistorie per jaar")
    sub.add_parser("params", help="gefitte parameters per competitie")
    sub.add_parser("selftest", help="reconstrueer eci_data en vergelijk")
    f = sub.add_parser("fixtures", help="reconstrueer voor een eigen wedstrijdlijst")
    f.add_argument("--csv", required=True)
    a = p.parse_args()

    if a.cmd == "coverage": cmd_coverage()
    elif a.cmd == "params": cmd_params()
    elif a.cmd == "selftest": cmd_selftest()
    else: cmd_fixtures(a.csv)


if __name__ == "__main__":
    main()