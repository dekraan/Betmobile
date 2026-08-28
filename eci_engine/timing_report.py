"""
timing_report.py

Wanneer kun je het beste inzetten?

De CLV-analyse liet zien dat picks gemiddeld +1,5% betere odds pakken dan de
closing line, maar dat slechts 41% de close verslaat. Dit script splitst dat
uit naar KOOPMOMENT: wat had je betaald op 72, 48, 24, 6 of 1 uur voor de
aftrap, en welk moment leverde de beste prijs op?

Waarom dit een andere vraag is dan de rest van het onderzoek: dit gaat niet
over "weet ECI iets" (beantwoord: nee), maar over "koop ik op het juiste
moment". Daar is geen model voor nodig - alleen prijsverloop, en dat staat
al in de snapshots.

Twee metrieken per venster:
- gem_odds       : de gemiddelde prijs die je op dat moment had gekregen
- edge_vs_close  : verwachte winst als de ge-devigde closing kans klopt

LET OP bij het lezen: een hogere prijs vroeg is alleen nuttig als de markt
daarna JOUW kant op beweegt. Beweegt hij de andere kant op, dan kocht je te
duur zonder het te merken - je had de weddenschap immers al.

Gebruik:
    python timing_report.py
    python timing_report.py --export-csv
    python timing_report.py --keep-all
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from clv_report import (
    load_picks,
    load_link,
    load_run_times,
    load_kickoffs,
    load_snapshots,
    build_closing,
    print_header,
    print_table,
    _selected,
)
from prob_calibration import compute_market_probs

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Koopmomenten in uren voor de aftrap.
WINDOWS = [72, 48, 24, 12, 6, 3, 1]

# Een snapshot telt alleen als hij dicht genoeg bij het gewenste moment ligt.
MAX_SLACK_HOURS = 6.0


# =====================================================================
# PRIJS OP EEN GEGEVEN MOMENT
# =====================================================================

def price_at_windows(snaps: pd.DataFrame, kickoffs: pd.DataFrame) -> pd.DataFrame:
    """
    Voor elke fixture en elk venster: de laatste snapshot die op of vóór dat
    moment viel.

    "Laatste vóór T-24u" is de prijs die je zou hebben gepakt als je 24 uur
    van tevoren had ingezet.
    """
    df = snaps.merge(kickoffs, on="fixture_id", how="inner")
    df["hours_before"] = (
        (df["kickoff_at"] - df["captured_at"]).dt.total_seconds() / 3600.0
    )
    df = df[df["hours_before"] >= 0]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for w in WINDOWS:
        part = df[df["hours_before"] >= w].copy()
        if part.empty:
            continue
        # Dichtst bij het venster = kleinste hours_before dat er nog boven ligt
        part = part.sort_values("hours_before").groupby("fixture_id", as_index=False).first()
        part = part[part["hours_before"] <= w + MAX_SLACK_HOURS]
        part["window_h"] = w
        rows.append(part[[
            "fixture_id", "window_h", "hours_before",
            "odds_home", "odds_draw", "odds_away",
        ]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_timing_frame(
    picks: pd.DataFrame,
    link: pd.DataFrame,
    windows: pd.DataFrame,
    closing: pd.DataFrame,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Koppel picks aan hun prijs op elk koopmoment en aan de closing line."""
    df = picks.copy()
    if dedupe:
        before = len(df)
        sort_col = "run_id" if "run_id" in df.columns else df.columns[0]
        df = (
            df.sort_values(sort_col)
            .drop_duplicates(subset=["match_id", "selection"], keep="first")
        )
        if before != len(df):
            print(f"[dedup] {before} -> {len(df)} unieke picks")

    df = df.merge(link, on="match_id", how="inner")
    df = df.merge(closing[["fixture_id", "close_home", "close_draw", "close_away"]],
                  on="fixture_id", how="inner")

    merged = df.merge(windows, on="fixture_id", how="inner",
                      suffixes=("_pick", "_win"))
    if merged.empty:
        return merged

    # Ge-devigde closing kans van de gekozen uitkomst = onze maatstaf.
    merged = compute_market_probs(
        merged,
        odds_cols=("close_home", "close_draw", "close_away"),
        out_cols=("p_close_home", "p_close_draw", "p_close_away"),
    )
    merged["p_close_sel"] = _selected(
        merged, "p_close_home", "p_close_draw", "p_close_away")

    # Prijs op dit koopmoment (kolommen komen uit de windows-tabel).
    merged["odds_at_window"] = _selected(
        merged, "odds_home_win", "odds_draw_win", "odds_away_win")
    merged["odds_close_sel"] = _selected(
        merged, "close_home", "close_draw", "close_away")

    merged["edge_vs_close"] = merged["p_close_sel"] * merged["odds_at_window"] - 1.0
    merged["clv_odds_pct"] = merged["odds_at_window"] / merged["odds_close_sel"] - 1.0
    merged["profit"] = np.where(
        merged["outcome"] == "WIN", merged["odds_at_window"] - 1.0, -1.0)
    return merged


# =====================================================================
# RAPPORT
# =====================================================================

def summarize_windows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, part in df.groupby("window_h", observed=True):
        edge = part["edge_vs_close"].dropna().to_numpy(float)
        n = len(edge)
        se = edge.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "uur_voor_aftrap": int(w),
            "picks": n,
            "gem_odds": float(part["odds_at_window"].mean()),
            "clv_odds_pct": float(part["clv_odds_pct"].mean()),
            "aandeel_beter_dan_close": float((part["clv_odds_pct"] > 0).mean()),
            "edge_vs_close": float(edge.mean()) if n else np.nan,
            "t_stat": float(edge.mean() / se) if se and se > 0 else np.nan,
            "roi_gerealiseerd": float(part["profit"].mean()),
        })
    return pd.DataFrame(rows).sort_values("uur_voor_aftrap", ascending=False)


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zelfde vergelijking, maar per competitieklasse.

    Hypothese: grote competities openen vroeg met ruime marges en scherpen
    daarna aan; dan zou vroeg kopen daar meer opleveren dan bij kleine
    competities. Vergelijkt bewust ALLEEN het vroegste en het laatste
    venster, zodat de tabel leesbaar blijft.
    """
    from prob_calibration import assign_competition_class
    from fit_calibration import CLASS_PATTERNS, DEFAULT_CLASS, _patterns_calib_stub

    stub = _patterns_calib_stub(CLASS_PATTERNS, DEFAULT_CLASS)
    d = df.copy()
    d["competition_class"] = assign_competition_class(d["competition"], stub)

    vroeg_h = max(WINDOWS)
    laat_h = min(WINDOWS)
    rows = []
    for (cls, w), part in d[d["window_h"].isin([vroeg_h, laat_h])].groupby(
        ["competition_class", "window_h"], observed=True
    ):
        rows.append({
            "klasse": cls,
            "venster": f"T-{int(w)}u",
            "picks": len(part),
            "gem_odds": float(part["odds_at_window"].mean()),
            "clv_odds_pct": float(part["clv_odds_pct"].mean()),
            "edge_vs_close": float(part["edge_vs_close"].mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Verschil vroeg vs laat per klasse: dit is de eigenlijke vraag.
    piv = out.pivot(index="klasse", columns="venster", values="clv_odds_pct")
    n_piv = out.pivot(index="klasse", columns="venster", values="picks")
    kolom_vroeg, kolom_laat = f"T-{vroeg_h}u", f"T-{laat_h}u"
    if kolom_vroeg in piv.columns and kolom_laat in piv.columns:
        piv["winst_van_vroeg_kopen"] = piv[kolom_vroeg] - piv[kolom_laat]
        piv["picks_vroeg"] = n_piv[kolom_vroeg]
        piv["picks_laat"] = n_piv[kolom_laat]
    return piv.reset_index()


def run_report(dedupe: bool = True, export_csv: bool = False) -> pd.DataFrame:
    print_header("BETMOBILE TIMING RAPPORT")
    print(
        "Vraag: op welk moment voor de aftrap kreeg je de beste prijs?\n"
        "Geen model nodig - alleen het prijsverloop uit de snapshots."
    )

    picks = load_picks()
    if picks.empty:
        print("Geen gesettelde picks.")
        return pd.DataFrame()

    link, _ = load_link()
    matched = picks.merge(link, on="match_id", how="inner")
    fixture_ids = sorted(set(int(x) for x in matched["fixture_id"].dropna()))
    if not fixture_ids:
        print("Geen picks gekoppeld aan een fixture.")
        return pd.DataFrame()

    kickoffs = load_kickoffs(fixture_ids)
    snaps = load_snapshots(fixture_ids)
    closing = build_closing(snaps, kickoffs)
    windows = price_at_windows(snaps, kickoffs)

    if windows.empty:
        print("Geen snapshots gevonden om koopmomenten mee te bepalen.")
        return pd.DataFrame()

    df = build_timing_frame(picks, link, windows, closing, dedupe=dedupe)
    if df.empty:
        print("Geen picks met bruikbaar prijsverloop.")
        return df

    summary = summarize_windows(df)
    print_table("PRIJS EN CLV PER KOOPMOMENT", summary)

    per_class = summarize_by_class(df)
    if not per_class.empty:
        print_table(
            "VROEG VS LAAT PER COMPETITIEKLASSE (verkorten grote competities meer?)",
            per_class,
        )
        print(
            "winst_van_vroeg_kopen > 0: in die klasse was de vroege prijs beter.\n"
            "Let op de aantallen: bij minder dan ~40 picks per cel is het verschil\n"
            "niet te onderscheiden van toeval."
        )

    print_header("HOE TE LEZEN")
    print(
        "gem_odds hoger bij vroege vensters = de prijs verkort richting de\n"
        "aftrap; dat is in jouw voordeel ALS je vroeg koopt.\n"
        "edge_vs_close is de geldmetriek: die moet > 0 zijn voor echte winst.\n"
        "Let op de kolom picks: verder terug in de tijd zijn er minder\n"
        "snapshots, dus vroege vensters rusten op minder waarnemingen."
    )

    # Welk venster gaf de hoogste edge? Alleen zinvol bij genoeg data.
    bruikbaar = summary[summary["picks"] >= 50]
    if not bruikbaar.empty:
        best = bruikbaar.sort_values("edge_vs_close", ascending=False).iloc[0]
        slecht = bruikbaar.sort_values("edge_vs_close").iloc[0]
        verschil = best["edge_vs_close"] - slecht["edge_vs_close"]
        print(
            f"\nbeste venster : T-{int(best['uur_voor_aftrap'])}u "
            f"(edge {best['edge_vs_close']:+.2%}, n={int(best['picks'])})\n"
            f"slechtste     : T-{int(slecht['uur_voor_aftrap'])}u "
            f"(edge {slecht['edge_vs_close']:+.2%}, n={int(slecht['picks'])})\n"
            f"verschil      : {verschil:+.2%} per weddenschap"
        )
        if verschil < 0.02:
            print(
                "Dat verschil is klein; het koopmoment maakt dan weinig uit en\n"
                "de gemeten volgorde kan toeval zijn."
            )

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p1 = EXPORT_DIR / f"timing_summary_{stamp}.csv"
        summary.to_csv(p1, index=False, encoding="utf-8-sig")
        print(f"\n[export] {p1}")
        if not per_class.empty:
            p3 = EXPORT_DIR / f"timing_by_class_{stamp}.csv"
            per_class.to_csv(p3, index=False, encoding="utf-8-sig")
            print(f"[export] {p3}")
        p2 = EXPORT_DIR / f"timing_detail_{stamp}.csv"
        df.to_csv(p2, index=False, encoding="utf-8-sig")
        print(f"[export] {p2}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Timing: beste koopmoment")
    p.add_argument("--keep-all", action="store_true",
                   help="Geen dedup naar eerste pick per match+selectie")
    p.add_argument("--export-csv", action="store_true")
    a = p.parse_args()
    run_report(dedupe=not a.keep_all, export_csv=a.export_csv)


if __name__ == "__main__":
    main()