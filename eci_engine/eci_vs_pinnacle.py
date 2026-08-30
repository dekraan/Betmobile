"""
eci_vs_pinnacle.py

TEST 5C - Is het ECI-effect uit test 5B echt, of mechanisch?

Test 5B vond: beweging ~ mkt_open + eci_prob geeft b(eci) = +0,0804 met
95%-interval [+0,0260, +0,1348]. Dat lijkt te zeggen dat ECI voorspelt waar
de markt naartoe beweegt.

MAAR ER IS EEN MECHANISCH ALTERNATIEF
    beweging  = p_close  - mkt_open
    afwijking = eci_prob - mkt_open
Beide bevatten -mkt_open. Elke ruis in de openingsprijs zit met hetzelfde
teken in de uitkomst EN in de verklarende variabele. Dat levert automatisch
een positieve coefficient op, zonder dat ECI iets weet. Controleren voor
mkt_open lost dit niet op: die controlevariabele IS de ruizige variabele.

DE BESLISSENDE TEST
Vervang ECI door een tweede MARKTmeting zonder ECI-inhoud: de ge-devigde
Pinnacle-openingskans, gemeten op hetzelfde tijdstip. Pinnacle weet per
definitie niets van ECI. Krijgt Pinnacle dezelfde coefficient, dan is het
effect mechanisch.

VIER MODELLEN, ZELFDE POPULATIE
    0  beweging ~ mkt_open
    1  beweging ~ mkt_open + eci_prob
    2  beweging ~ mkt_open + pin_open
    3  beweging ~ mkt_open + eci_prob + pin_open      <- de beslissende

Model 3 scheidt de gevallen:
    ECI overleeft naast Pinnacle          -> ECI heeft eigen informatie
    ECI verdwijnt, Pinnacle blijft        -> het was marktinformatie
    beide verdwijnen                      -> het was collineariteit/ruis
    beide blijven                         -> beide dragen bij

CRUCIAAL: alle vier de modellen draaien op EXACT dezelfde wedstrijden (die
waarvoor ook een Pinnacle-opening bestaat). Anders vergelijk je populaties
in plaats van modellen. Model 1 wordt daarom opnieuw geschat en kan licht
afwijken van de +0,0804 uit test 5B; dat verschil is populatie, geen model.

STANDAARDFOUTEN
Naast klassiek ook HC3 (heteroskedasticiteit) en cluster-robuust per
competitie. Wedstrijden uit dezelfde competitie delen marktstructuur en zijn
niet onafhankelijk. Alle drie worden gerapporteerd; het strengste interval
telt.

PREREGISTRATIE (vastgelegd voor de eerste run)
  Het ECI-effect uit 5B overleeft alleen als in MODEL 3 de coefficient op
  eci_prob positief blijft EN het cluster-robuuste interval nul uitsluit.
  Verdwijnt eci_prob zodra pin_open erbij komt, dan wordt 5B verworpen als
  mechanisch/marktinformatie.
  Ook overleven is GEEN bevestiging: de hypothese komt uit deze data.

Gebruik:
    python eci_vs_pinnacle.py
    python eci_vs_pinnacle.py --export-csv
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

MIN_EERSTE_H = 48.0
MAX_LAATSTE_H = 12.0
PINNACLE_ID = 4
# Hoe ver mag de Pinnacle-snapshot van het modelrun-moment liggen?
PIN_SLACK_H = 6.0


# =====================================================================
# STANDAARDFOUTEN
# =====================================================================

def _fit(y: np.ndarray, X: np.ndarray):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return beta, resid, XtX_inv


def se_klassiek(X, resid, XtX_inv):
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    return np.sqrt(np.diag(s2 * XtX_inv))


def se_hc3(X, resid, XtX_inv):
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    h = np.clip(h, 0, 0.9999)
    w = (resid / (1.0 - h)) ** 2
    meat = X.T @ (X * w[:, None])
    return np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))


def se_cluster(X, resid, XtX_inv, groepen):
    n, k = X.shape
    codes = pd.Categorical(groepen).codes
    g = len(np.unique(codes))
    meat = np.zeros((k, k))
    for c in np.unique(codes):
        m = codes == c
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    corr = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return np.sqrt(np.diag(XtX_inv @ (corr * meat) @ XtX_inv)), g


def regressie(df: pd.DataFrame, termen: list[str], cluster_col: str = "competition"):
    y = df["beweging"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df))]
                        + [df[t].to_numpy(dtype=float) for t in termen])
    namen = ["constante"] + termen
    beta, resid, XtX_inv = _fit(y, X)
    se_c = se_klassiek(X, resid, XtX_inv)
    se_h = se_hc3(X, resid, XtX_inv)
    se_g, n_groepen = se_cluster(X, resid, XtX_inv, df[cluster_col])

    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / tss if tss > 0 else np.nan

    out = pd.DataFrame({
        "term": namen,
        "coef": beta,
        "se_klas": se_c,
        "se_hc3": se_h,
        "se_clus": se_g,
        "clus_laag": beta - 1.96 * se_g,
        "clus_hoog": beta + 1.96 * se_g,
        "sig_clus": np.abs(beta) > 1.96 * se_g,
    })
    out.attrs["r2"] = r2
    out.attrs["n"] = len(df)
    out.attrs["clusters"] = n_groepen
    return out


# =====================================================================
# DATA
# =====================================================================

def load_first_run() -> pd.DataFrame:
    q = f"""
        WITH spans AS (
            SELECT match_id,
                   MAX(hours_to_kickoff) AS eerste_h,
                   MIN(hours_to_kickoff) AS laatste_h
            FROM public.model_match_snapshots
            WHERE hours_to_kickoff IS NOT NULL
            GROUP BY match_id
        ),
        net AS (
            SELECT match_id, eerste_h, laatste_h FROM spans
            WHERE eerste_h >= {MIN_EERSTE_H} AND laatste_h <= {MAX_LAATSTE_H}
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
        SELECT e.*, n.eerste_h, n.laatste_h
        FROM eerste e JOIN net n USING (match_id)
        WHERE e.odds_home IS NOT NULL AND e.odds_draw IS NOT NULL
          AND e.odds_away IS NOT NULL
          AND e.prob_home IS NOT NULL AND e.prob_away IS NOT NULL
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    print(f"[load] nette reeksen: {len(df)}")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home = df["prob_home"] >= df["prob_away"]
    df["eci_side"] = np.where(home, "HOME", "AWAY")
    df["eci_prob"] = np.where(home, df["prob_home"], df["prob_away"])
    df["odds_open"] = np.where(home, df["odds_home"], df["odds_away"])
    df = compute_market_probs(df, out_cols=("mkt_home", "mkt_draw", "mkt_away"))
    df["mkt_open"] = np.where(home, df["mkt_home"], df["mkt_away"])
    df["afwijking"] = df["eci_prob"] - df["mkt_open"]
    return df


def add_closing(df: pd.DataFrame) -> pd.DataFrame:
    link, _ = load_link()
    df = df.copy()
    df["fixture_id"] = df["match_id"].map(link.set_index("match_id")["fixture_id"])
    ids = df["fixture_id"].dropna().astype(int).unique().tolist()
    closing = build_closing(load_snapshots(ids), load_kickoffs(ids))
    closing = compute_market_probs(
        closing,
        odds_cols=("close_home", "close_draw", "close_away"),
        out_cols=("p_close_home", "p_close_draw", "p_close_away"),
    )
    df = df.merge(
        closing[["fixture_id", "p_close_home", "p_close_away", "kickoff_at"]],
        on="fixture_id", how="left",
    )
    p_close = np.where(df["eci_side"] == "HOME",
                       df["p_close_home"], df["p_close_away"])
    df["p_close"] = p_close
    df["beweging"] = (p_close - df["mkt_open"]) * df["odds_open"]
    return df


def add_pinnacle(df: pd.DataFrame) -> pd.DataFrame:
    """Pinnacle-openingskans op hetzelfde moment als de eerste modelrun.

    Het modelrun-moment is kickoff_at - hours_to_kickoff. We zoeken de
    Pinnacle-snapshot die daar het dichtst bij ligt en er niet na komt,
    binnen PIN_SLACK_H uur.
    """
    ids = df["fixture_id"].dropna().astype(int).unique().tolist()
    if not ids:
        df["pin_open"] = np.nan
        return df
    lijst = ",".join(str(i) for i in ids)
    q = f"""
        SELECT fixture_id, captured_at,
               MAX(odd) FILTER (WHERE label = 'Home') AS ph,
               MAX(odd) FILTER (WHERE label = 'Draw') AS pd_,
               MAX(odd) FILTER (WHERE label = 'Away') AS pa
        FROM public.odds_values_snapshots
        WHERE bookmaker_id = {PINNACLE_ID}
          AND market_key = '1x2'
          AND fixture_id IN ({lijst})
        GROUP BY fixture_id, captured_at
        HAVING MAX(odd) FILTER (WHERE label = 'Home') IS NOT NULL
           AND MAX(odd) FILTER (WHERE label = 'Draw') IS NOT NULL
           AND MAX(odd) FILTER (WHERE label = 'Away') IS NOT NULL
    """
    with db_engine().connect() as conn:
        pin = pd.read_sql(q, conn)
    if pin.empty:
        print("[pin] geen Pinnacle-snapshots gevonden.")
        df["pin_open"] = np.nan
        return df
    pin["captured_at"] = pd.to_datetime(pin["captured_at"], utc=True, errors="coerce")
    print(f"[pin] {len(pin)} Pinnacle-snapshots voor {pin['fixture_id'].nunique()} fixtures")

    pin = compute_market_probs(
        pin, odds_cols=("ph", "pd_", "pa"),
        out_cols=("pin_home", "pin_draw", "pin_away"),
    )

    df = df.copy()
    df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True, errors="coerce")
    df["run_moment"] = df["kickoff_at"] - pd.to_timedelta(
        df["hours_to_kickoff"].astype(float), unit="h")

    links = df[["match_id", "fixture_id", "run_moment"]].dropna().copy()
    links["fixture_id"] = links["fixture_id"].astype("int64")
    links = links.sort_values("run_moment")
    pin_s = pin.sort_values("captured_at").copy()
    pin_s["fixture_id"] = pin_s["fixture_id"].astype("int64")
    gekoppeld = pd.merge_asof(
        links, pin_s,
        left_on="run_moment", right_on="captured_at",
        by="fixture_id", direction="nearest",
        tolerance=pd.Timedelta(hours=PIN_SLACK_H),
    )
    gekoppeld["pin_gap_h"] = (
        gekoppeld["captured_at"] - gekoppeld["run_moment"]
    ).dt.total_seconds().abs() / 3600.0

    df = df.merge(
        gekoppeld[["match_id", "pin_home", "pin_away", "pin_gap_h"]],
        on="match_id", how="left",
    )
    df["pin_open"] = np.where(df["eci_side"] == "HOME",
                              df["pin_home"], df["pin_away"])
    n = int(df["pin_open"].notna().sum())
    print(f"[pin] gekoppeld voor {n} van {len(df)} wedstrijden "
          f"(mediane afstand {df['pin_gap_h'].median():.2f} uur)")
    return df


# =====================================================================
# UITVAL
# =====================================================================

def uitval(volledig: pd.DataFrame, behouden: pd.DataFrame, reden: str) -> None:
    weg = volledig[~volledig["match_id"].isin(behouden["match_id"])]
    print(f"\n=== UITVAL: {reden} ({len(weg)} weg, {len(behouden)} over) ===")
    if weg.empty:
        print("Geen uitval.")
        return
    rijen = []
    for naam, sub in [("behouden", behouden), ("weggevallen", weg)]:
        rijen.append({
            "groep": naam, "n": len(sub),
            "gem_odds": sub["odds_open"].mean() if "odds_open" in sub else np.nan,
            "eci_prob": sub["eci_prob"].mean() if "eci_prob" in sub else np.nan,
            "afwijking": sub["afwijking"].mean() if "afwijking" in sub else np.nan,
            "eerste_h": sub["eerste_h"].mean(),
            "is_pick": sub["is_pick"].mean() if "is_pick" in sub else np.nan,
        })
    print(pd.DataFrame(rijen).round(4).to_string(index=False))
    top = (weg["competition"].value_counts().head(8)
           / volledig["competition"].value_counts()).dropna().sort_values(ascending=False)
    print("\nGrootste uitval per competitie (aandeel van die competitie):")
    print(top.head(8).round(3).to_string())


# =====================================================================
# MAIN
# =====================================================================

def toon(titel: str, res: pd.DataFrame) -> None:
    print(f"\n=== {titel} ===")
    print(f"n = {res.attrs['n']}, clusters = {res.attrs['clusters']}, "
          f"R2 = {res.attrs['r2']:.4f}")
    print(res.round(5).to_string(index=False))


def oordeel(m3: pd.DataFrame) -> str:
    e = m3[m3["term"] == "eci_prob"].iloc[0]
    p = m3[m3["term"] == "pin_open"].iloc[0]
    eci_ok = bool(e["sig_clus"]) and e["coef"] > 0
    pin_ok = bool(p["sig_clus"]) and p["coef"] > 0
    if eci_ok and not pin_ok:
        return ("ECI OVERLEEFT: eci_prob blijft significant naast Pinnacle, "
                "Pinnacle niet. Opvallend. GEEN bevestiging - hypothese komt "
                "uit deze data en moet op verse data getoetst worden.")
    if eci_ok and pin_ok:
        return ("BEIDE DRAGEN BIJ: ECI en Pinnacle voegen allebei iets toe "
                "bovenop de openingsprijs. Consistent met 'tweede meting van "
                "dezelfde kans', dus mechanisch NIET uitgesloten.")
    if pin_ok and not eci_ok:
        return ("5B VERWORPEN: zodra een tweede marktmeting erbij komt "
                "verdwijnt het ECI-effect. Het was marktinformatie, geen ECI.")
    return ("5B VERWORPEN: geen van beide predictoren overleeft "
            "cluster-robuuste standaardfouten.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args()

    print("\n=== TEST 5C: ECI VERSUS PINNACLE ALS PLACEBO ===")
    print("Overleeft het ECI-effect uit 5B een tweede marktmeting?\n")

    basis = prepare(load_first_run())
    vol = add_closing(basis)
    met_clv = vol[vol["beweging"].notna()].copy()
    uitval(vol, met_clv, "geen closing-data")

    met_pin = add_pinnacle(met_clv)
    df = met_pin[met_pin["pin_open"].notna()].copy()
    uitval(met_clv, df, "geen Pinnacle-opening")

    if len(df) < 200:
        print(f"\nSTOP: slechts {len(df)} wedstrijden met zowel closing als "
              "Pinnacle. Te weinig voor een zinvolle vergelijking.")
        return

    print(f"\nAnalysepopulatie: {len(df)} wedstrijden, "
          f"{df['competition'].nunique()} competities")
    print(f"corr(eci_prob, mkt_open) = {df['eci_prob'].corr(df['mkt_open']):+.4f}")
    print(f"corr(pin_open, mkt_open) = {df['pin_open'].corr(df['mkt_open']):+.4f}")
    print(f"corr(eci_prob, pin_open) = {df['eci_prob'].corr(df['pin_open']):+.4f}")

    modellen = {
        "MODEL 0 - alleen openingsprijs": ["mkt_open"],
        "MODEL 1 - prijs + ECI":          ["mkt_open", "eci_prob"],
        "MODEL 2 - prijs + Pinnacle":     ["mkt_open", "pin_open"],
        "MODEL 3 - prijs + ECI + Pinnacle": ["mkt_open", "eci_prob", "pin_open"],
    }
    resultaten = {naam: regressie(df, t) for naam, t in modellen.items()}
    for naam, res in resultaten.items():
        toon(naam, res)

    m3 = resultaten["MODEL 3 - prijs + ECI + Pinnacle"]
    print(f"\n>>> OORDEEL: {oordeel(m3)}")

    print("\nLezen: vergelijk de coefficient op eci_prob in model 1 en model 3.")
    print("Zakt hij richting nul zodra Pinnacle erbij komt, dan mat 5B geen ECI")
    print("maar een tweede schatting van dezelfde onderliggende kans.")

    if args.export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        df.to_csv(EXPORT_DIR / f"test5c_data_{stamp}.csv", index=False)
        for naam, res in resultaten.items():
            slug = naam.split(" - ")[0].replace(" ", "").lower()
            res.to_csv(EXPORT_DIR / f"test5c_{slug}_{stamp}.csv", index=False)
        print(f"[export] naar {EXPORT_DIR}")


if __name__ == "__main__":
    main()