# -*- coding: utf-8 -*-
"""
test_price_break.py

H-PRICE-1 — is de prijsbreuk rond odds 1.50 echt?

AANLEIDING
De isotone EV-curve in tier_definition.json vindt één breekpunt tussen odds
1,518 en 1,585: eronder -1,73%, erboven -6,90%. Het huidige regelvenster
(min_odds 1.50, max_odds 2.50) ligt daarmee vrijwel volledig aan de dure
kant. Voordat daar iets aan verandert moet vaststaan dat het verschil geen
toeval is.

PREREGISTRATIE — vastgelegd voordat er een uitkomst is bekeken
  H0  Het rendement per inzet op uitkomsten onder odds 1.50 is gelijk aan
      dat op uitkomsten tussen 1.50 en 2.50.
  H1  Onder 1.50 is het rendement hoger.

  Toets       OLS van profit op de indicator cheap = (odds < 1.50),
              binnen de populatie odds < 2.50.
  Fouten      Cluster-robuust per competitie, kritieke waarde t(G-1).
              Wedstrijden binnen een competitie delen marktstructuur; de
              drie uitkomsten van één wedstrijd zijn bovendien onderling
              afhankelijk en vallen binnen hetzelfde cluster.
  Primair     Alleen de hold-out helft (eci_date > SPLIT_DATE). De breuk is
              op de trainhelft gevonden, dus daar toetsen zou circulair zijn.
  Secundair   Volledige periode, puur ter informatie.
  Verwerpen   als het cluster-robuuste 95%-interval van cheap in de
              HOLD-OUT nul uitsluit.

  Drempel     1.50 is bewust NIET het datagekozen breekpunt (~1,52-1,58)
              maar de bestaande configuratiewaarde. Daarmee is de grens niet
              uit dezelfde curve geplukt die hem moet bewijzen.

BEPERKINGEN — hoort in elk oordeel mee
  - odds komen uit oddspedia_unibet_backbone, geschreven als
    'oddspedia_presented': niet één vaste bookmaker en geen vast tijdstip
    t.o.v. aftrap. Het NIVEAU van de curve is daarmee onzeker; de VORM
    (favourite-longshot) is dat veel minder.
  - home_win_pct komt uit eci_data, een upsert-tabel met de huidige stand.
    Zolang archive/check_eci_lookahead.py niet is gedraaid, is niet
    uitgesloten dat daar informatie van na de wedstrijd in zit. Dat raakt
    de ECI-kolommen, niet de odds en de uitslag - de primaire toets hier
    gebruikt alleen odds en uitslag en staat daar dus los van.

READ-ONLY. Draait uitsluitend SELECT-queries.

Gebruik
-------
    python test_price_break.py
    python test_price_break.py --selftest      # estimator controleren, geen DB
    python test_price_break.py --max-odds 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SOURCE = "public.eci_oddspedia_matches"
SPLIT_DATE = "2026-01-24"          # zelfde split als tier_definition.json
CHEAP_BREAK = 1.50                 # preregistreerde grens
DEFAULT_MAX_ODDS = 2.50            # bovengrens huidige regelvenster


# =====================================================================
# DATA
# =====================================================================

def load_matches() -> pd.DataFrame:
    from sqlalchemy import create_engine
    from betmobile_settings import DB_DSN

    q = f"""
        SELECT eci_date, competition,
               odds_home, odds_draw, odds_away,
               home_win_pct, draw_pct, away_win_pct,
               value_home, value_away,
               rating_diff, result
        FROM {SOURCE}
        WHERE result IS NOT NULL
          AND odds_home IS NOT NULL
          AND odds_draw IS NOT NULL
          AND odds_away IS NOT NULL
    """
    engine = create_engine(DB_DSN)
    with engine.begin() as conn:
        df = pd.read_sql(q, conn)
    df["eci_date"] = pd.to_datetime(df["eci_date"], errors="coerce")
    return df.dropna(subset=["eci_date"])


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Eén rij per mogelijke weddenschap: drie per wedstrijd."""
    frames = []
    for side, res in [("home", "HOME"), ("draw", "DRAW"), ("away", "AWAY")]:
        odds = pd.to_numeric(df[f"odds_{side}"], errors="coerce")
        hit = (df["result"] == res).astype(float)
        frames.append(pd.DataFrame({
            "eci_date": df["eci_date"].values,
            "competition": df["competition"].values,
            "side": side,
            "odds": odds.values,
            "hit": hit.values,
            "profit": np.where(hit.values > 0, odds.values - 1.0, -1.0),
            "eci_prob": pd.to_numeric(
                df[f"{'draw' if side == 'draw' else side + '_win'}_pct"],
                errors="coerce").values,
            "rating_diff": pd.to_numeric(df["rating_diff"], errors="coerce").values,
        }))
    long = pd.concat(frames, ignore_index=True)
    return long[(long["odds"] > 1.01) & long["odds"].notna()].copy()


# =====================================================================
# CLUSTER-ROBUUSTE OLS
# =====================================================================

def cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """
    OLS met cluster-robuuste sandwich-fouten en t(G-1) als kritieke waarde.

    Geen scipy/statsmodels-regressie: de sandwich staat hier expliciet, zodat
    zichtbaar is dat er over CLUSTERS gesommeerd wordt en niet over rijen.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    codes = pd.factorize(pd.Series(groups))[0]
    G = int(codes.max()) + 1

    meat = np.zeros((k, k))
    for g in range(G):
        m = codes == g
        Xg, eg = X[m], resid[m]
        s = Xg.T @ eg
        meat += np.outer(s, s)

    # Eindige-steekproefcorrectie, gangbaar bij clusterfouten
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = c * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))

    tcrit = float(stats.t.ppf(0.975, max(G - 1, 1)))
    tstat = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), max(G - 1, 1)))

    return {
        "beta": beta, "se": se, "t": tstat, "p": pval,
        "tcrit": tcrit, "G": G, "n": n,
        "lo": beta - tcrit * se, "hi": beta + tcrit * se,
    }


def run_test(long: pd.DataFrame, label: str, max_odds: float) -> dict | None:
    sub = long[long["odds"] < max_odds].copy()
    sub["cheap"] = (sub["odds"] < CHEAP_BREAK).astype(float)

    n_cheap = int(sub["cheap"].sum())
    n_dear = int(len(sub) - n_cheap)
    print(f"\n--- {label} ---")
    print(f"populatie: odds < {max_odds}   n = {len(sub)}   "
          f"(< {CHEAP_BREAK}: {n_cheap} | >= {CHEAP_BREAK}: {n_dear})")

    if n_cheap < 50 or n_dear < 50:
        print("te weinig waarnemingen aan één kant; geen toets.")
        return None

    X = np.column_stack([np.ones(len(sub)), sub["cheap"].to_numpy()])
    r = cluster_ols(sub["profit"].to_numpy(), X, sub["competition"].to_numpy())

    print(f"clusters (competities): {r['G']}   kritieke waarde t({r['G']-1}) = {r['tcrit']:.3f}")
    print(f"  rendement >= {CHEAP_BREAK} : {r['beta'][0]*100:+.2f}%")
    print(f"  verschil (cheap)     : {r['beta'][1]*100:+.2f} pp"
          f"   [{r['lo'][1]*100:+.2f} ; {r['hi'][1]*100:+.2f}]"
          f"   t = {r['t'][1]:.2f}   p = {r['p'][1]:.4f}")
    print(f"  rendement <  {CHEAP_BREAK} : {(r['beta'][0]+r['beta'][1])*100:+.2f}%")

    excl = (r["lo"][1] > 0) or (r["hi"][1] < 0)
    print(f"  interval sluit nul uit: {'JA' if excl else 'nee'}")
    return r


def describe_by_bucket(long: pd.DataFrame, max_odds: float) -> None:
    bins = [1.01, 1.2, 1.35, 1.5, 1.65, 1.8, 2.0, 2.5]
    sub = long[long["odds"] < max_odds].copy()
    sub["bucket"] = pd.cut(sub["odds"], bins=bins)
    g = sub.groupby("bucket", observed=True).agg(
        n=("profit", "size"), gem_odds=("odds", "mean"),
        hitrate=("hit", "mean"), rendement=("profit", "mean"))
    g["rendement_pct"] = (g["rendement"] * 100).round(2)
    print("\n--- beschrijvend per prijsklasse (hold-out) ---")
    print(g[["n", "gem_odds", "hitrate", "rendement_pct"]].round(3).to_string())


# =====================================================================
# SELFTEST
# =====================================================================

def selftest() -> None:
    """Estimator op gesimuleerde data met een bekend effect van +5 pp."""
    rng = np.random.default_rng(7)
    rows = []
    for comp in range(30):
        for _ in range(400):
            cheap = rng.random() < 0.3
            odds = rng.uniform(1.15, 1.49) if cheap else rng.uniform(1.5, 2.5)
            # werkelijke kans zo gezet dat EV -2% (cheap) resp. -7% is
            ev = -0.02 if cheap else -0.07
            p = (1 + ev) / odds
            hit = rng.random() < p
            rows.append((comp, odds, float(cheap), odds - 1 if hit else -1.0))
    d = pd.DataFrame(rows, columns=["competition", "odds", "cheap", "profit"])
    X = np.column_stack([np.ones(len(d)), d["cheap"].to_numpy()])
    r = cluster_ols(d["profit"].to_numpy(), X, d["competition"].to_numpy())
    print("SELFTEST — verwacht verschil +5.00 pp")
    print(f"  gevonden: {r['beta'][1]*100:+.2f} pp "
          f"[{r['lo'][1]*100:+.2f} ; {r['hi'][1]*100:+.2f}]  "
          f"t = {r['t'][1]:.2f}  clusters = {r['G']}")
    ok = r["lo"][1] < 0.05 < r["hi"][1]
    print(f"  ware waarde binnen interval: {'JA' if ok else 'NEE'}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max-odds", type=float, default=DEFAULT_MAX_ODDS)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    print("H-PRICE-1 — prijsbreuk rond odds 1.50")
    print(f"bron: {SOURCE}   split: {SPLIT_DATE}   grens: {CHEAP_BREAK}")

    df = load_matches()
    print(f"\nwedstrijden met uitslag: {len(df)}")

    long = to_long(df)
    split = pd.Timestamp(SPLIT_DATE)
    holdout = long[long["eci_date"] > split]
    train = long[long["eci_date"] <= split]
    print(f"weddenschappen: {len(long)}   "
          f"(train t/m {SPLIT_DATE}: {len(train)} | hold-out: {len(holdout)})")

    run_test(holdout, "PRIMAIR — hold-out", a.max_odds)
    run_test(long, "SECUNDAIR — volledige periode (ter informatie)", a.max_odds)
    describe_by_bucket(holdout, a.max_odds)

    print("\nLet op: een verschil bewijst dat de PRIJSKLASSE ertoe doet, niet")
    print("dat de picks erdoor winstgevend worden. Beide kanten blijven")
    print("negatief; dit gaat over hoeveel je per inzet weggeeft.")


if __name__ == "__main__":
    main()