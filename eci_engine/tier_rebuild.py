"""
tier_rebuild.py

Bouwt de pick-tiers opnieuw op, op basis van GESCHATTE VERWACHTINGSWAARDE
in plaats van op segmenten uit de auto-discovery.

WAAROM DE OUDE TIERS NIET WERKEN
1. Ze hangen aan segmenten die gevonden zijn door honderden combinaties af
   te zoeken op dezelfde data waarop ze daarna beoordeeld werden.
2. Ze optimaliseren op "vaak goed voorspeld" in plaats van op "winstgevend".
   Wedstrijden met een duidelijke favoriet zijn goed voorspelbaar EN slecht
   betaald; die twee heffen elkaar op.
   Meetbaar gevolg: A- claimt gemiddeld 78% en A+ 69% - de tiers zijn niet
   eens geordend op kans.

HOE DE NIEUWE TIERS WERKEN
    geschatte EV = het historische rendement van weddenschappen in
                   dezelfde prijsklasse (odds-bucket)
    tier         = vaste EV-grenzen

Waarom direct op rendement en niet via een kanscorrectie: de proportionele
devig-methode kent systematisch te weinig kans toe aan favorieten. Meet je
"bias" op ge-devigde kansen en reken je daarna af tegen de echte odds, dan
tel je dat artefact als winst mee - dat gaf in de eerste versie een
fantoom-EV van +0,7% waar -5,9% uitkwam. Door direct het gerealiseerde
rendement per prijsklasse te meten, zit de marge er automatisch in en is
er geen devig-aanname nodig.

EERLIJKE VERWACHTING: ook de beste tier komt negatief uit, want de marge
(~8,7% bij Bet365) is groter dan de bias. Wat je krijgt is een betrouwbare
RANGORDE - bruikbaar voor een shortlist en voor staking, niet voor winst.

Gebruik:
    python tier_rebuild.py
    python tier_rebuild.py --export-csv
    python tier_rebuild.py --with-eci      # test of ECI-instemming iets toevoegt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from db import refresh_source_views
from fit_calibration import (
    DEFAULT_SCHEMA,
    DEFAULT_SOURCE,
    TRAIN_FRAC,
    load_match_frame,
    prepare_match_frame,
    print_header,
    print_table,
)

TIER_DIR = OUTPUT_DIR / "calibration"
TIER_DIR.mkdir(parents=True, exist_ok=True)
TIER_CONFIG_PATH = TIER_DIR / "tier_definition.json"

# Prijsklassen waarop het rendement gemeten wordt (op de odds zelf).
from shared_buckets import ODDS_BINS_FINE as ODDS_BINS, ODDS_LABELS_FINE as ODDS_LABELS

# Krimp naar het algemene gemiddelde: een prijsklasse met weinig
# waarnemingen krijgt niet zijn volle (toevallige) rendement toebedeeld.
SHRINK_K = 300

# Vaste EV-grenzen. Bewust ROND en vooraf gekozen, niet geoptimaliseerd op
# de uitkomst - anders introduceren we dezelfde fout als bij de oude tiers.
# Vaste, vooraf gekozen EV-grenzen. Bewust ronde getallen, niet bijgesteld
# op basis van hoe de testset eruitziet - dat laatste zou precies de fout
# zijn die de oude tiers onbruikbaar maakte.
#
# Eerder geprobeerd: grenzen afleiden uit trainset-kwantielen, zodat elke
# tier een streefaandeel kreeg. Dat gaf een SLECHTER resultaat (67% van de
# overgangen liep af in plaats van 100%, en een tier met 139 wedstrijden
# waar niets over te zeggen valt). Bewaard als les: fijnmaziger indelen dan
# de data toelaat, maakt de indeling slechter, niet beter.
TIER_EDGES = [
    ("A+", -0.03),   # EV >= -3%: de minst slechte prijsklassen
    ("A", -0.05),
    ("B", -0.08),
    ("C", -0.12),
    ("D", -np.inf),
]

MIN_BUCKET = 100

# Tiers met minder waarnemingen tellen niet mee in het rangorde-oordeel:
# een tier met een handvol wedstrijden zegt niets en zou het oordeel
# domineren door toevalsuitschieters.
MIN_TIER_N = 100


# =====================================================================
# BIAS-CURVE
# =====================================================================

def fit_ev_curve(train: pd.DataFrame) -> dict:
    """
    Meet per prijsklasse het gerealiseerde rendement.

    Elke wedstrijd levert drie mogelijke weddenschappen (H/D/A). Voor elk
    daarvan: wat leverde 1 unit op? Het gemiddelde per odds-bucket is de
    directe schatting van de EV in die prijsklasse - inclusief marge, zonder
    devig-aanname.

    Kleine buckets worden richting het algemene gemiddelde gekrompen
    (empirical-Bayes-achtig), zodat een toevallige uitschieter geen eigen
    tier krijgt.
    """
    rows = []
    for i, side in enumerate(["home", "draw", "away"]):
        odds = train[f"odds_{side}"].to_numpy(float)
        hit = (train["y_idx"].to_numpy(int) == i).astype(float)
        rows.append(pd.DataFrame({
            "odds": odds,
            "profit": np.where(hit > 0, odds - 1.0, -1.0),
            "hit": hit,
        }))
    long = pd.concat(rows, ignore_index=True).dropna()
    long = long[long["odds"] > 1.01]
    long["bucket"] = pd.cut(long["odds"], bins=ODDS_BINS, labels=ODDS_LABELS)

    overall = float(long["profit"].mean())

    grp = (
        long.groupby("bucket", observed=True)
        .agg(n=("profit", "size"), gem_odds=("odds", "mean"),
             ruw_rendement=("profit", "mean"), hitrate=("hit", "mean"))
        .reset_index()
    )
    # Krimp: hoe kleiner de bucket, hoe dichter bij het algemene gemiddelde.
    w = grp["n"] / (grp["n"] + SHRINK_K)
    grp["ev"] = w * grp["ruw_rendement"] + (1 - w) * overall

    curve = [
        {"bucket": str(b), "low": float(lo), "high": (None if np.isinf(hi) else float(hi)),
         "ev": float(ev), "n": int(n)}
        for b, lo, hi, ev, n in zip(
            grp["bucket"], ODDS_BINS[:-1], ODDS_BINS[1:], grp["ev"], grp["n"])
    ]
    return {"bins": ODDS_BINS, "labels": ODDS_LABELS, "curve": curve,
            "table": grp, "overall": overall}


def lookup_ev(odds: np.ndarray, curve: list[dict], overall: float) -> np.ndarray:
    """Zoek per weddenschap de geschatte EV op basis van de prijsklasse."""
    out = np.full(len(odds), overall, dtype=float)
    for c in curve:
        hi = np.inf if c["high"] is None else c["high"]
        m = (odds > c["low"]) & (odds <= hi)
        out[m] = c["ev"]
    return out


# =====================================================================
# EV EN TIERS
# =====================================================================

def compute_ev(df: pd.DataFrame, curve: list[dict], overall: float,
               eci_bonus: float = 0.0) -> pd.DataFrame:
    """
    Beoordeel per wedstrijd de weddenschap die je in de praktijk zou doen:
    die op de MARKTFAVORIET.

    Waarom niet "de weddenschap met de hoogste geschatte EV"? Omdat de EV
    alleen van de prijsklasse afhangt; dan zou het script overal dezelfde
    prijsklasse kiezen en vallen alle wedstrijden in één tier. De favoriet
    is bovendien wat de engine feitelijk selecteert.

    eci_bonus: opslag op de EV wanneer ECI dezelfde uitkomst als favoriet
    ziet. Standaard 0; met --with-eci toont de validatie of het helpt.
    """
    df = df.copy()
    odds = df[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    eci = df[["mdl_home", "mdl_draw", "mdl_away"]].to_numpy(float)
    mkt = df[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(float)

    sel = mkt.argmax(axis=1)
    rows = np.arange(len(df))
    sel_odds = odds[rows, sel]

    ev = lookup_ev(sel_odds, curve, overall)
    agrees = eci.argmax(axis=1) == sel
    if eci_bonus:
        ev = ev + np.where(agrees, eci_bonus, 0.0)

    df["sel_idx"] = sel
    df["selection"] = np.array(["HOME", "DRAW", "AWAY"])[sel]
    df["sel_odds"] = sel_odds
    df["sel_mkt_prob"] = mkt[rows, sel]
    df["sel_ev"] = ev
    df["sel_hit"] = (sel == df["y_idx"].to_numpy(int)).astype(float)
    df["sel_profit"] = np.where(df["sel_hit"] > 0, sel_odds - 1.0, -1.0)
    df["eci_agrees"] = agrees
    return df


def assign_tier(ev: np.ndarray) -> np.ndarray:
    out = np.full(len(ev), "D", dtype=object)
    for name, edge in TIER_EDGES:
        out = np.where((ev >= edge) & (out == "D"), name, out)
    return out


def tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = assign_tier(df["sel_ev"].to_numpy(float))
    order = {name: i for i, (name, _) in enumerate(TIER_EDGES)}
    rows = []
    for tier, part in df.groupby("tier", observed=True):
        profit = part["sel_profit"].to_numpy(float)
        n = len(profit)
        se = profit.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "tier": tier,
            "n": n,
            "gem_odds": float(part["sel_odds"].mean()),
            "verwachte_ev": float(part["sel_ev"].mean()),
            "werkelijke_roi": float(profit.mean()),
            "roi_lo": float(profit.mean() - 1.96 * se) if n > 1 else np.nan,
            "roi_hi": float(profit.mean() + 1.96 * se) if n > 1 else np.nan,
            "hitrate": float(part["sel_hit"].mean()),
        })
    out = pd.DataFrame(rows)
    out["telt_mee"] = out["n"] >= MIN_TIER_N
    out["_o"] = out["tier"].map(order)
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def validate_curve(test: pd.DataFrame, curve: list[dict], overall: float) -> tuple[pd.DataFrame, bool, str]:
    """
    De doorslaggevende toets: voorspelt de EV-curve (gefit op train) het
    werkelijke rendement per prijsklasse op de TESTSET?

    Waarom op bucketniveau en niet per weddenschap: het rendement van een
    losse weddenschap heeft een spreiding van ongeveer 1 unit, waardoor een
    toets per weddenschap zelfs bij 14.000 waarnemingen te weinig
    onderscheidend vermogen heeft. Per prijsklasse middelt die ruis uit.

    Verwacht bij een werkende curve: helling rond 1 (voorspeld rendement
    komt overeen met werkelijk) en een positieve samenhang.
    """
    rows = []
    for i, side in enumerate(["home", "draw", "away"]):
        odds = test[f"odds_{side}"].to_numpy(float)
        hit = (test["y_idx"].to_numpy(int) == i).astype(float)
        rows.append(pd.DataFrame({
            "odds": odds,
            "profit": np.where(hit > 0, odds - 1.0, -1.0),
        }))
    long = pd.concat(rows, ignore_index=True).dropna()
    long = long[long["odds"] > 1.01]
    long["bucket"] = pd.cut(long["odds"], bins=ODDS_BINS, labels=ODDS_LABELS)
    long["voorspeld"] = lookup_ev(long["odds"].to_numpy(float), curve, overall)

    grp = (
        long.groupby("bucket", observed=True)
        .agg(n=("profit", "size"), gem_odds=("odds", "mean"),
             voorspeld=("voorspeld", "mean"), werkelijk=("profit", "mean"))
        .reset_index()
    )
    grp["verschil"] = grp["werkelijk"] - grp["voorspeld"]
    grp = grp[grp["n"] >= MIN_TIER_N]

    if len(grp) < 3:
        return grp, False, "te weinig prijsklassen met data"

    w = grp["n"].to_numpy(float)
    x = grp["voorspeld"].to_numpy(float)
    y = grp["werkelijk"].to_numpy(float)
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    slope = float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))
    corr = float(np.corrcoef(x, y)[0, 1])

    ok = bool(slope > 0.3 and corr > 0.5)
    note = (
        f"helling {slope:+.2f} (1.00 = perfect), samenhang {corr:+.2f} -> "
        f"{'curve voorspelt het rendement' if ok else 'curve voorspelt niet betrouwbaar'}"
    )
    return grp, ok, note


# =====================================================================
# HOOFDPROGRAMMA
# =====================================================================

def run(source: str, schema: str, train_frac: float, refresh: bool,
        export_csv: bool, with_eci: bool, eci_bonus: float,
        df: pd.DataFrame | None = None) -> dict:
    print_header("TIERS OPNIEUW OPBOUWEN OP GESCHATTE EV")
    print(
        "Oude tiers: segmenten uit auto-discovery -> optimaliseren op\n"
        "'vaak goed voorspeld', wat samenvalt met lage odds.\n"
        "Nieuwe tiers: geschatte EV = (marktkans + bias) x odds - 1."
    )

    if df is None:
        if refresh:
            try:
                refresh_source_views()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] refresh mislukt: {exc}")
        df = prepare_match_frame(load_match_frame(source, schema))
    else:
        df = prepare_match_frame(df)
    df = df.reset_index(drop=True)

    split = df["date_dt"].quantile(train_frac)
    train, test = df[df["date_dt"] <= split], df[df["date_dt"] > split]
    print(f"[split] {pd.Timestamp(split).date()} | train={len(train)} | test={len(test)}")

    # ---- 1. Rendement per prijsklasse op de TRAIN-helft ----
    fit = fit_ev_curve(train)
    tbl = fit["table"][["bucket", "n", "gem_odds", "hitrate", "ruw_rendement", "ev"]].copy()
    tbl["bucket"] = tbl["bucket"].astype(str)
    print_table("1. RENDEMENT PER PRIJSKLASSE (op trainset)", tbl)
    print(
        f"algemeen gemiddelde: {fit['overall']:+.2%} (dit is ongeveer de marge)\n"
        "ruw_rendement = wat die prijsklasse opleverde; ev = na krimp richting\n"
        "het gemiddelde, zodat kleine buckets geen eigen tier krijgen.\n"
        "Geen devig-aanname: de marge zit al in de odds verwerkt."
    )

    print_header("1b. TIERGRENZEN (vast, vooraf gekozen)")
    for name, edge in TIER_EDGES:
        grens = "rest" if np.isinf(edge) else f"EV >= {edge:+.4f}"
        print(f"  {name:>3}: {grens}")

    bonus = eci_bonus if with_eci else 0.0
    if with_eci:
        print(f"\n[eci] ECI-instemming krijgt een opslag van {bonus:+.3f} op de kans.")

    # ---- 2. Tiers op de TESTSET (eerlijke beoordeling) ----
    test_ev = compute_ev(test, fit["curve"], fit["overall"], eci_bonus=bonus)
    summary = tier_summary(test_ev)
    print_table("2. NIEUWE TIERS OP DE TESTSET", summary)

    val_tbl, ok, msg = validate_curve(test, fit["curve"], fit["overall"])
    print_table("2b. VALIDATIE: VOORSPELD VS WERKELIJK PER PRIJSKLASSE (testset)", val_tbl)
    vals = summary[summary["telt_mee"]]["werkelijke_roi"].to_numpy(float)
    dalend = float((np.diff(vals) <= 0).mean()) if len(vals) > 1 else np.nan
    print(
        f"\n{msg}\n"
        f"(ter info: {dalend:.0%} van de tier-overgangen loopt af; met ROI per\n"
        f"weddenschap is een rangorde-toets te ruizig om iets te bewijzen)\n"
        f"-> {'BRUIKBAAR voor rangschikken' if ok else 'NIET BRUIKBAAR'}"
    )
    t_stat = float("nan")

    # ---- 3. Vergelijking met de simpelste alternatieven ----
    base_fav = test[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(float).argmax(axis=1)
    odds_mat = test[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    y = test["y_idx"].to_numpy(int)
    fav_profit = np.where(base_fav == y, odds_mat[np.arange(len(test)), base_fav] - 1.0, -1.0)

    grote = summary[summary["telt_mee"]]
    best_tier = grote.iloc[0] if not grote.empty else summary.iloc[0]
    print_header("3. VERGELIJKING OP DEZELFDE TESTSET")
    print(
        f"blind op marktfavoriet : n={len(fav_profit)}, roi={fav_profit.mean():+.2%}\n"
        f"beste nieuwe tier ({best_tier['tier']})  : n={int(best_tier['n'])}, "
        f"roi={best_tier['werkelijke_roi']:+.2%} "
        f"[{best_tier['roi_lo']:+.2%}, {best_tier['roi_hi']:+.2%}]"
    )

    # ---- 4. Bevriezen ----
    version = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config = {
        "version": version,
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "source": f"{schema}.{source}",
        "split_date": str(pd.Timestamp(split).date()),
        "method": "realised_ev_by_price",
        "ev_curve": fit["curve"],
        "overall_ev": fit["overall"],
        "tier_edges": [[n, (None if np.isinf(e) else float(e))] for n, e in TIER_EDGES],
        "eci_bonus": bonus,
        "validation": {
            "ordering_works": bool(ok),
            "note": msg,
            "t_stat": (None if np.isnan(t_stat) else float(t_stat)),
            "test_n": int(len(test)),
        },
    }
    with open(TIER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n[export] tier-definitie: {TIER_CONFIG_PATH} (versie {version})")

    print_header("HOE TE LEZEN")
    print(
        "verwachte_ev is wat het model vooraf denkt; werkelijke_roi is wat er\n"
        "uitkwam. Liggen die dicht bij elkaar EN loopt de rij netjes af, dan is\n"
        "de rangorde bruikbaar. Let op: ook A+ is waarschijnlijk negatief - de\n"
        "marge is groter dan de bias. Dit rangschikt, het maakt niet winstgevend."
    )

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, table in [("tier_bias_curve", tbl), ("tier_summary_test", summary)]:
            path = TIER_DIR / f"{name}_{stamp}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[export] {path}")

    return {"summary": summary, "ordering_works": ok, "config": config}


def main() -> None:
    p = argparse.ArgumentParser(description="Tiers opnieuw opbouwen op EV")
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--export-csv", action="store_true")
    p.add_argument("--with-eci", action="store_true",
                   help="Geef ECI-instemming een kleine opslag en kijk of het helpt")
    p.add_argument("--eci-bonus", type=float, default=0.014,
                   help="Opslag bij ECI-instemming (default 0.014 = het gemeten residu)")
    a = p.parse_args()
    run(a.source, a.schema, a.train_frac, not a.no_refresh, a.export_csv,
        a.with_eci, a.eci_bonus)


if __name__ == "__main__":
    main()