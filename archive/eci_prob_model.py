"""
eci_prob_model.py
=================

Toetst of de ECI-wedstrijdkansen te reconstrueren zijn uit het ratingverschil.

Het model komt van de methodology-pagina: wedstrijdprestatie is normaal
verdeeld rond het ratingverschil plus thuisvoordeel, gelijkspel als het
verschil binnen een drempel valt (ordered probit).

    z = (ECI_thuis - ECI_uit + H) / sigma
    P(thuis)  = 1 - Phi(tau - z)
    P(uit)    = Phi(-tau - z)
    P(gelijk) = rest

Parameters gefit op 1.897 wedstrijden uit de match-odds feed van augustus
2026. Dit script toetst ze op eci_data, dat teruggaat tot februari 2025 —
dus een echte out-of-sample controle, en meteen een test of ECI zijn
parameters tussentijds bijstelt.

Wat het NIET kan: de Euro Player Index. Die zit wel in de gepubliceerde
kansen maar is een gesloten bron van Remiqz. Dat is de bovengrens van
wat reconstructie kan bereiken.

READ-ONLY. Gebruikt DB_CONFIG uit betmobile_settings.

Gebruik
-------
    python eci_prob_model.py validate
    python eci_prob_model.py refit
    python eci_prob_model.py refit --by-competition
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
    from scipy.optimize import minimize
except ImportError:
    sys.exit("scipy ontbreekt. Installeer met: pip install scipy")

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 ontbreekt. Installeer met: pip install psycopg2-binary")

try:
    from betmobile_settings import DB_CONFIG
except ImportError:
    sys.exit("betmobile_settings.py niet gevonden. Draai vanuit je Betmobile-map.")


# Gefit op de match-odds feed, augustus 2026.
H0, SIGMA0, TAU0 = 182.9, 1026.2, 0.3467


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def probs(H, sigma, tau, gap):
    z = (gap + H) / sigma
    p_away = norm.cdf(-tau - z)
    p_home = 1.0 - norm.cdf(tau - z)
    return p_home, 1.0 - p_home - p_away, p_away


def load() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(readonly=True, autocommit=True)
    df = pd.read_sql("""
        SELECT date, home_team, away_team, competition,
               home_rating, away_rating,
               home_win_pct, draw_pct, away_win_pct
        FROM eci_data
        WHERE home_rating IS NOT NULL
          AND away_rating IS NOT NULL
          AND home_win_pct IS NOT NULL
          AND COALESCE(is_excluded, false) = false
    """, conn)
    conn.close()

    # Ratings staan als tekst in de database; expliciet casten.
    for col in ("home_rating", "away_rating"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_rating", "away_rating", "date"])

    # De kansen kunnen als fractie of als percentage opgeslagen zijn.
    if df["home_win_pct"].max() > 1.5:
        for col in ("home_win_pct", "draw_pct", "away_win_pct"):
            df[col] = df[col] / 100.0

    df["gap"] = df.home_rating - df.away_rating
    return df.reset_index(drop=True)


def score(df, params, label=""):
    ph, pd_, pa = probs(*params, df.gap.values)
    err = ph - df.home_win_pct.values
    mae = np.abs(err).mean() * 100
    med = np.median(np.abs(err)) * 100
    r2 = 1 - np.var(err) / np.var(df.home_win_pct.values)
    if label:
        print(f"  {label:<26} n={len(df):>6,}  MAE {mae:5.2f}pp  "
              f"mediaan {med:5.2f}pp  R2 {r2:6.3f}")
    return mae, med, r2


def fit(df, x0=(H0, SIGMA0, TAU0)):
    gap = df.gap.values
    pH, pD, pA = (df.home_win_pct.values, df.draw_pct.values,
                  df.away_win_pct.values)

    def loss(p):
        if p[1] <= 1 or p[2] <= 0:
            return 1e9
        ph, pd_, pa = probs(p[0], p[1], p[2], gap)
        return np.mean((ph - pH) ** 2 + (pd_ - pD) ** 2 + (pa - pA) ** 2)

    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"maxiter": 20000, "fatol": 1e-14, "xatol": 1e-9})
    return res.x


# ---------------------------------------------------------------------------

def run_validate() -> None:
    df = load()
    hr("Bereik van de data")
    print(f"  {len(df):,} wedstrijden, {df.date.min():%Y-%m-%d} tot {df.date.max():%Y-%m-%d}")
    print(f"  {df.competition.nunique()} competities")

    hr("Model met de parameters uit augustus 2026")
    print(f"  H={H0}  sigma={SIGMA0}  tau={TAU0}\n")
    score(df, (H0, SIGMA0, TAU0), "alles")

    print()
    for period, sub in df.groupby(df.date.dt.to_period("Q")):
        if len(sub) >= 50:
            score(sub, (H0, SIGMA0, TAU0), str(period))

    hr("Parameters per kwartaal opnieuw gefit")
    print("  Blijven H, sigma en tau stabiel, dan past ECI zijn model niet aan.\n")
    print(f"  {'periode':<12} {'n':>6}  {'H':>7} {'sigma':>8} {'tau':>7}  {'MAE':>6}")
    for period, sub in df.groupby(df.date.dt.to_period("Q")):
        if len(sub) < 100:
            continue
        p = fit(sub)
        mae, _, _ = score(sub, p)
        print(f"  {str(period):<12} {len(sub):>6,}  {p[0]:7.1f} {p[1]:8.1f} "
              f"{p[2]:7.4f}  {mae:5.2f}pp")

    hr("Per competitie")
    print("  Grote competities: kansen volgen het ratingverschil.")
    print("  Kleine competities: de Euro Player Index doet de rest.\n")
    rows = []
    for comp, sub in df.groupby("competition"):
        if len(sub) < 40:
            continue
        mae, med, r2 = score(sub, (H0, SIGMA0, TAU0))
        rows.append((comp, len(sub), mae, r2))
    rows.sort(key=lambda r: r[2])

    print(f"  {'competitie':<26} {'n':>6}  {'MAE':>7}  {'R2':>6}")
    for comp, n, mae, r2 in rows[:10]:
        print(f"  {str(comp)[:26]:<26} {n:>6,}  {mae:6.2f}pp  {r2:6.3f}")
    print("  ...")
    for comp, n, mae, r2 in rows[-10:]:
        print(f"  {str(comp)[:26]:<26} {n:>6,}  {mae:6.2f}pp  {r2:6.3f}")

    hr("CONCLUSIE")
    mae, med, r2 = score(df, (H0, SIGMA0, TAU0))
    good = [r for r in rows if r[2] < 5.0]
    print(f"  Mediane fout over alles: {med:.2f} procentpunt")
    print(f"  Competities met MAE < 5pp: {len(good)} van {len(rows)}")
    print()
    print("  Voor die competities kun je historische ECI-kansen reconstrueren")
    print("  uit ratings_history.jsonl, terug tot juli 2007.")
    print("  Voor de rest niet: daar zit de EPI in, en die is niet publiek.")


def run_refit(by_competition: bool) -> None:
    df = load()
    hr("Opnieuw fitten op eci_data")

    p = fit(df)
    print(f"  H     = {p[0]:.1f}   (was {H0})")
    print(f"  sigma = {p[1]:.1f}   (was {SIGMA0})")
    print(f"  tau   = {p[2]:.4f}   (was {TAU0})\n")
    score(df, p, "nieuwe parameters")
    score(df, (H0, SIGMA0, TAU0), "oude parameters")

    p500 = probs(*p, np.array([500.0]))[0][0]
    print(f"\n  ijkpunt gap=500 -> {p500*100:.1f}% thuiswinst "
          f"(methodology noemt 66%)")

    if not by_competition:
        print("\n  Draai met --by-competition voor parameters per competitie.")
        return

    hr("Parameters per competitie")
    print("  Wijkt H sterk af, dan heeft die competitie een ander thuisvoordeel.\n")
    print(f"  {'competitie':<26} {'n':>6}  {'H':>7} {'sigma':>8} {'tau':>7}  {'MAE':>6}")
    for comp, sub in sorted(df.groupby("competition"), key=lambda x: -len(x[1])):
        if len(sub) < 150:
            continue
        pc = fit(sub, x0=p)
        mae, _, _ = score(sub, pc)
        print(f"  {str(comp)[:26]:<26} {len(sub):>6,}  {pc[0]:7.1f} {pc[1]:8.1f} "
              f"{pc[2]:7.4f}  {mae:5.2f}pp")


def main() -> None:
    parser = argparse.ArgumentParser(description="ECI-kansmodel uit ratingverschil")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="toets de gefitte parameters op eci_data")
    p_re = sub.add_parser("refit", help="fit de parameters opnieuw op eci_data")
    p_re.add_argument("--by-competition", action="store_true",
                      help="ook per competitie fitten")
    args = parser.parse_args()

    if args.command == "validate":
        run_validate()
    else:
        run_refit(args.by_competition)


if __name__ == "__main__":
    main()