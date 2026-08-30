"""
eci_market_movement.py

TEST 5B - Voorspelt de ECI-afwijking de LATERE MARKTBEWEGING?

Test 5A (selector_ladder.py) was negatief: ECI-kans en rating gap als
selector leveren geen rendement. Maar in ladder C viel iets op - hoe verder
ECI boven de markt zat, hoe slechter de prijs zich daarna ontwikkelde, en
spiegelbeeldig aan de andere kant. Deze test onderzoekt dat patroon.

WAAROM DIT EEN ANDERE VRAAG IS
Niet "voorspelt ECI de uitslag" (vier keer nee), maar "voorspelt ECI waar de
markt naartoe beweegt". Marktbeweging heeft veel minder ruis dan uitslagen,
dus dit convergeert bij een fractie van de steekproef.

DE UITKOMSTMAAT
    edge_shin = mkt_prob(kant) x odds_open - 1     verwacht, bij open
    clv       = p_close(kant)  x odds_open - 1     gerealiseerd, bij close
    beweging  = clv - edge_shin

Beide op DEZELFDE odds_open, dus de marge valt weg en wat overblijft is puur
de verschuiving van de marktkans tussen open en close. Positief = de markt
bewoog naar de gekozen kant toe.

TIJDSCONSTRUCTIE (geen lookahead)
Selectie, ECI-kans, marktkans en prijs komen ALLEMAAL uit de eerste run.
Alleen de close is toekomstinformatie. Verantwoord omdat gemeten is dat ECI
tussen eerste en laatste run praktisch stilstaat: correlatie 0,9977,
gemiddelde verschuiving 0,0036, 0,83% zijwissels. Die 16 zijwissels blijven
er bewust in - de eerste-run-keuze is leidend, anders selecteer je achteraf.

Populatie: alleen wedstrijden met eerste zicht >= 48u en laatste zicht <= 12u
voor de aftrap (de "nette reeks", n ~ 1177). Puur timingcriteria, dus
onafhankelijk van de uitkomst.

DE FORMELE TOETS IS EEN REGRESSIE, GEEN PLACEBO-LADDER
Oorspronkelijk plan was een placebo-ladder op het competitiegemiddelde. Dat
werkt niet: bij dezelfde gekozen kant is de beweging per wedstrijd identiek
voor signaal en placebo, dus het verschil per wedstrijd is exact nul. Wat
verschilt is alleen WELKE wedstrijden in welke drempelbak vallen, en daar
bestaat geen gepaarde toets voor.

De vraag is dus een regressievraag:

    beweging ~ b0 + b1*mkt_prob + b2*eci_prob

b1 vangt het prijseffect (mean reversion: staat een prijs hoog, dan zakt hij
daarna). b2 is wat ECI daar bovenop toevoegt. Alleen b2 telt.

Equivalente tweede specificatie ter controle:

    beweging ~ b0 + b1*mkt_prob + b2*afwijking      (afwijking = eci - mkt)

PREREGISTRATIE (vastgelegd voor de eerste run)
  Positief bewijs vereist dat b2 in BEIDE specificaties hetzelfde teken heeft
  en dat het 95%-interval nul uitsluit. Het teken van b1 doet niet mee - dat
  is de markt, niet ECI. De ladders zijn beschrijvend en tellen niet mee.

  EN OOK BIJ EEN SIGNIFICANTE b2 IS DIT GEEN BEVESTIGING. Ladder C is in
  deze data ontdekt en wordt hier in dezelfde data getoetst. Wat hier
  uitkomt is een scherper geformuleerde hypothese, te bevestigen op verse
  data vanaf het vriespunt (29 augustus 2026).

Gebruik:
    python eci_market_movement.py
    python eci_market_movement.py --export-csv
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from db import db_engine
from prob_calibration import compute_market_probs
from clv_report import load_link, load_kickoffs, load_snapshots, build_closing

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_EERSTE_H = 48.0   # eerste zicht minstens zoveel uur voor aftrap
MAX_LAATSTE_H = 12.0  # laatste zicht binnen zoveel uur voor aftrap

LADDER_DREMPELS = [0.00, 0.02, 0.05, 0.08, 0.12]
LADDER_DREMPELS_NEG = [0.00, -0.02, -0.05]


# =====================================================================
# DATA
# =====================================================================

def load_first_run() -> pd.DataFrame:
    """Toestand bij de EERSTE modelrun, alleen voor de nette reeksen."""
    q = f"""
        WITH spans AS (
            SELECT match_id,
                   MAX(hours_to_kickoff) AS eerste_h,
                   MIN(hours_to_kickoff) AS laatste_h,
                   COUNT(*) AS n_runs
            FROM public.model_match_snapshots
            WHERE hours_to_kickoff IS NOT NULL
            GROUP BY match_id
        ),
        net AS (
            SELECT match_id, eerste_h, laatste_h, n_runs
            FROM spans
            WHERE eerste_h >= {MIN_EERSTE_H}
              AND laatste_h <= {MAX_LAATSTE_H}
        ),
        eerste AS (
            SELECT DISTINCT ON (match_id)
                   match_id, run_id, date, competition,
                   odds_home, odds_draw, odds_away,
                   prob_home, prob_draw, prob_away,
                   rating_gap, is_pick, hours_to_kickoff
            FROM public.model_match_snapshots
            ORDER BY match_id, run_id
        )
        SELECT e.*, n.eerste_h, n.laatste_h, n.n_runs
        FROM eerste e
        JOIN net n USING (match_id)
        WHERE e.odds_home IS NOT NULL
          AND e.odds_draw IS NOT NULL
          AND e.odds_away IS NOT NULL
          AND e.prob_home IS NOT NULL
          AND e.prob_away IS NOT NULL
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    print(f"[load] nette reeksen met eerste-run-data: {len(df)}")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    home = df["prob_home"] >= df["prob_away"]
    df["eci_side"] = np.where(home, "HOME", "AWAY")
    df["eci_prob"] = np.where(home, df["prob_home"], df["prob_away"])
    df["odds_open"] = np.where(home, df["odds_home"], df["odds_away"])

    df = compute_market_probs(df, out_cols=("mkt_home", "mkt_draw", "mkt_away"))
    df["mkt_prob"] = np.where(home, df["mkt_home"], df["mkt_away"])

    df["afwijking"] = df["eci_prob"] - df["mkt_prob"]
    df["edge_shin"] = df["mkt_prob"] * df["odds_open"] - 1.0
    return df


def add_closing(df: pd.DataFrame) -> pd.DataFrame:
    """Closing-kansen erbij; beweging = clv - edge_shin, beide op odds_open."""
    link, _ = load_link()
    lk = link.set_index("match_id")["fixture_id"]
    df = df.copy()
    df["fixture_id"] = df["match_id"].map(lk)

    ids = df["fixture_id"].dropna().astype(int).unique().tolist()
    if not ids:
        raise RuntimeError("geen gekoppelde fixtures")

    closing = build_closing(load_snapshots(ids), load_kickoffs(ids))
    if closing.empty:
        raise RuntimeError("geen closing-snapshots")

    closing = compute_market_probs(
        closing,
        odds_cols=("close_home", "close_draw", "close_away"),
        out_cols=("p_close_home", "p_close_draw", "p_close_away"),
    )
    df = df.merge(
        closing[["fixture_id", "p_close_home", "p_close_away",
                 "close_captured_at", "kickoff_at"]],
        on="fixture_id", how="left",
    )

    df["p_close"] = np.where(df["eci_side"] == "HOME",
                             df["p_close_home"], df["p_close_away"])
    df["clv"] = df["p_close"] * df["odds_open"] - 1.0
    df["beweging"] = df["clv"] - df["edge_shin"]

    # Sanity: beweging moet ook gelijk zijn aan (p_close - mkt_prob) * odds_open
    controle = (df["p_close"] - df["mkt_prob"]) * df["odds_open"]
    afw = (df["beweging"] - controle).abs().max()
    print(f"[check] beweging-identiteit max afwijking: {afw:.2e} (hoort ~0)")

    n = int(df["beweging"].notna().sum())
    print(f"[clv] beweging berekend voor {n} van {len(df)} wedstrijden")
    return df[df["beweging"].notna()].copy()


# =====================================================================
# REGRESSIE (formele toets)
# =====================================================================

def ols(y: np.ndarray, X: np.ndarray, namen: list[str]) -> pd.DataFrame:
    """Kleinste kwadraten met klassieke standaardfouten."""
    X = np.column_stack([np.ones(len(X)), X])
    namen = ["constante"] + namen
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame({
        "term": namen,
        "coef": beta,
        "se": se,
        "ci_laag": beta - 1.96 * se,
        "ci_hoog": beta + 1.96 * se,
        "sluit_nul_uit": (np.abs(beta) > 1.96 * se),
    })


def run_regressies(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = df["beweging"].to_numpy(dtype=float)

    spec1 = ols(y,
                df[["mkt_prob", "eci_prob"]].to_numpy(dtype=float),
                ["mkt_prob", "eci_prob"])
    spec2 = ols(y,
                df[["mkt_prob", "afwijking"]].to_numpy(dtype=float),
                ["mkt_prob", "afwijking"])
    # Referentie: alleen de prijs. Als spec1/spec2 hier niets aan toevoegen,
    # is ECI overbodig.
    spec0 = ols(y, df[["mkt_prob"]].to_numpy(dtype=float), ["mkt_prob"])
    return {"0. alleen prijs": spec0,
            "1. prijs + ECI-kans": spec1,
            "2. prijs + afwijking": spec2}


def toets(specs: dict[str, pd.DataFrame]) -> str:
    r1 = specs["1. prijs + ECI-kans"]
    r2 = specs["2. prijs + afwijking"]
    b1 = r1[r1["term"] == "eci_prob"].iloc[0]
    b2 = r2[r2["term"] == "afwijking"].iloc[0]
    zelfde_teken = np.sign(b1["coef"]) == np.sign(b2["coef"])
    beide_sig = bool(b1["sluit_nul_uit"]) and bool(b2["sluit_nul_uit"])
    if zelfde_teken and beide_sig:
        richting = "positief" if b1["coef"] > 0 else "negatief"
        return (f"SIGNAAL ({richting}): b2 in beide specificaties significant "
                "en met hetzelfde teken. GEEN bevestiging - hypothese is in "
                "deze data ontdekt en moet op verse data getoetst worden.")
    reden = []
    if not zelfde_teken:
        reden.append("tegenstrijdig teken tussen specificaties")
    if not beide_sig:
        reden.append("niet in beide specificaties significant")
    return "GEEN SIGNAAL: " + " en ".join(reden)


# =====================================================================
# BESCHRIJVEND
# =====================================================================

def _ci95(s: pd.Series) -> float:
    s = s.dropna()
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan


def ladder(df: pd.DataFrame, drempels: list[float], omlaag: bool = False) -> pd.DataFrame:
    rows = []
    for label, sub in [("alles", df)] + [
        (f"afwijking {'<=' if omlaag else '>='} {t:+.2f}",
         df[df["afwijking"] <= t] if omlaag else df[df["afwijking"] >= t])
        for t in drempels
    ]:
        if sub.empty:
            continue
        rows.append({
            "drempel": label,
            "n": len(sub),
            "gem_odds": sub["odds_open"].mean(),
            "eci_prob": sub["eci_prob"].mean(),
            "mkt_prob": sub["mkt_prob"].mean(),
            "afwijking": sub["afwijking"].mean(),
            "edge_shin": sub["edge_shin"].mean(),
            "clv": sub["clv"].mean(),
            "beweging": sub["beweging"].mean(),
            "beweging_ci": _ci95(sub["beweging"]),
            "beweging_med": sub["beweging"].median(),
            "pos_pct": (sub["beweging"] > 0).mean(),
        })
    return pd.DataFrame(rows)


def stratificatie(df: pd.DataFrame) -> pd.DataFrame:
    """Transparante tegenhanger van de regressie: binnen prijsklasse
    vergelijken we lage, midden en hoge afwijking. Als het patroon alleen
    tussen prijsklassen bestaat en niet erbinnen, was het de prijs."""
    d = df.copy()
    d["prijsklasse"] = pd.cut(d["odds_open"],
                              [1.0, 1.5, 1.8, 2.2, 3.0, np.inf],
                              labels=["<1.50", "1.50-1.80", "1.80-2.20",
                                      "2.20-3.00", "3.00+"], right=False)
    d["afw_tertiel"] = d.groupby("prijsklasse", observed=True)["afwijking"] \
        .transform(lambda s: pd.qcut(s, 3, labels=["laag", "midden", "hoog"],
                                     duplicates="drop"))
    out = (d.groupby(["prijsklasse", "afw_tertiel"], observed=True)
             .agg(n=("beweging", "size"),
                  gem_odds=("odds_open", "mean"),
                  afwijking=("afwijking", "mean"),
                  beweging=("beweging", "mean"),
                  ci=("beweging", _ci95))
             .reset_index())
    return out


def toon(titel: str, df: pd.DataFrame) -> None:
    print(f"\n=== {titel} ===")
    if df is None or df.empty:
        print("Geen data.")
        return
    v = df.copy()
    for c in v.columns:
        if pd.api.types.is_float_dtype(v[c]):
            v[c] = v[c].round(5)
    print(v.to_string(index=False))


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args()

    print("\n=== TEST 5B: VOORSPELT ECI DE LATERE MARKTBEWEGING? ===")
    print("Formele toets = regressie. De ladders zijn beschrijvend.")
    print("Ook een significante uitkomst is GEEN bevestiging: dit patroon is")
    print("in deze data ontdekt. Bevestiging vereist verse data.\n")

    df = add_closing(prepare(load_first_run()))

    print(f"\nn = {len(df)}")
    print(f"beweging: gemiddeld {df['beweging'].mean():+.5f}, "
          f"mediaan {df['beweging'].median():+.5f}, "
          f"sd {df['beweging'].std():.5f}")
    print(f"corr(afwijking, mkt_prob) = {df['afwijking'].corr(df['mkt_prob']):+.4f}")
    print(f"corr(afwijking, odds_open) = {df['afwijking'].corr(df['odds_open']):+.4f}")
    print(f"corr(eci_prob, mkt_prob)  = {df['eci_prob'].corr(df['mkt_prob']):+.4f}")

    specs = run_regressies(df)
    print("\n########## FORMELE TOETS ##########")
    for naam, res in specs.items():
        toon(f"REGRESSIE {naam}", res)
    print(f"\n>>> OORDEEL: {toets(specs)}")

    print("\n########## BESCHRIJVEND ##########")
    toon("LADDER - ECI boven de markt", ladder(df, LADDER_DREMPELS))
    toon("LADDER - ECI onder de markt", ladder(df, LADDER_DREMPELS_NEG, omlaag=True))
    toon("STRATIFICATIE - afwijking BINNEN prijsklasse", stratificatie(df))

    print("\nLezen: als de beweging binnen elke prijsklasse vlak is over de")
    print("tertielen, kwam het ladderpatroon van de prijs en niet van ECI.")

    if args.export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        df.to_csv(EXPORT_DIR / f"test5b_data_{stamp}.csv", index=False)
        for naam, res in specs.items():
            slug = naam.split(". ")[1].replace(" ", "_").replace("+", "en")
            res.to_csv(EXPORT_DIR / f"test5b_reg_{slug}_{stamp}.csv", index=False)
        print(f"[export] naar {EXPORT_DIR}")


if __name__ == "__main__":
    main()