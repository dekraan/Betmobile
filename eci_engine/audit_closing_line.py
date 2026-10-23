# -*- coding: utf-8 -*-
"""
audit_closing_line.py

Is de "closing" in het CLV-rapport wel een closing?

AANLEIDING
De picks verslaan de markt in ELKE afwijkingsbucket. Dat is onaannemelijk:
als het in elke bucket werkt, werkt het waarschijnlijk nergens en meet je
iets anders dan je denkt. De eerste verdachte is de closing-proxy.

clv_report.py definieert die als "de laatste Bet365-snapshot VOOR de
aftrap" (build_closing). Dat is alleen een closing als er ook echt kort
voor de aftrap gemeten is. Wordt er twee keer per dag gescrapet, dan kan
die "closing" tien uur oud zijn - en dan meet edge_vs_close niet de
marktbeweging maar het verschil tussen twee bijna gelijktijdige metingen
van dezelfde prijs.

WAT DIT SCRIPT DOET
  A  hoe lang VOOR de aftrap ligt de closing-snapshot, in uren
  B  hoeveel snapshots zijn er in het laatste etmaal en het laatste uur
  C  hoe vaak is de closing-odd exact gelijk aan de genomen odd
  D  hangt de gemeten CLV af van hoe goed de closing-proxy is

D is de beslissende. Is de CLV alleen positief waar de closing ver van de
aftrap ligt, dan is de CLV een meetartefact. Is hij vlak over alle banden,
dan is de proxy niet de verklaring en moet de oorzaak elders liggen.

Dit script IMPORTEERT uit clv_report.py in plaats van de logica na te
bouwen. Een audit van een nabouw bewijst niets over het origineel.

READ-ONLY. Draait alleen SELECT-queries, via de bestaande loaders.

Gebruik (vanuit eci_engine/)
---------------------------
    python audit_closing_line.py
    python audit_closing_line.py --export-csv
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from prob_calibration import compute_market_probs
from clv_report import (
    load_picks, load_link, load_run_times,
    load_kickoffs, load_snapshots, build_closing,
)

EXPORT_DIR = OUTPUT_DIR / "research"

# Banden voor "hoe dicht zit de closing op de aftrap"
LEAD_BINS = [-0.001, 1, 3, 6, 12, 24, 48, np.inf]
LEAD_LABELS = ["< 1u", "1-3u", "3-6u", "6-12u", "12-24u", "24-48u", "> 48u"]


def kop(t: str) -> None:
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


# =====================================================================
# A + B — KWALITEIT VAN DE CLOSING-PROXY
# =====================================================================

def audit_proxy(snaps: pd.DataFrame, kickoffs: pd.DataFrame,
                closing: pd.DataFrame) -> pd.DataFrame:
    kop("A — hoe oud is de closing-snapshot bij aftrap?")

    c = closing.copy()
    c["lead_h"] = (c["kickoff_at"] - c["close_captured_at"]).dt.total_seconds() / 3600.0
    c = c[c["lead_h"].notna()]

    q = c["lead_h"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    print(f"fixtures met een closing: {len(c)}")
    print(f"  mediaan   {q[0.50]:7.2f} uur voor aftrap")
    print(f"  p25 / p75 {q[0.25]:7.2f} / {q[0.75]:.2f}")
    print(f"  p10 / p90 {q[0.10]:7.2f} / {q[0.90]:.2f}")

    c["lead_band"] = pd.cut(c["lead_h"], bins=LEAD_BINS, labels=LEAD_LABELS)
    verdeling = c["lead_band"].value_counts().reindex(LEAD_LABELS).fillna(0).astype(int)
    print("\nverdeling:")
    for label, n in verdeling.items():
        aandeel = n / len(c) if len(c) else 0
        print(f"  {label:>8}  {n:6d}  {aandeel:6.1%}  {'#' * int(aandeel * 40)}")

    kop("B — meetdichtheid vlak voor de aftrap")
    s = snaps.merge(kickoffs, on="fixture_id", how="inner")
    s["h_voor"] = (s["kickoff_at"] - s["captured_at"]).dt.total_seconds() / 3600.0
    s = s[s["h_voor"] >= 0]
    per_fix = s.groupby("fixture_id").agg(
        in_48u=("h_voor", lambda x: int((x <= 48).sum())),
        in_24u=("h_voor", lambda x: int((x <= 24).sum())),
        in_6u=("h_voor", lambda x: int((x <= 6).sum())),
        in_1u=("h_voor", lambda x: int((x <= 1).sum())),
    )
    print(f"snapshots per fixture (mediaan):")
    for col in ["in_48u", "in_24u", "in_6u", "in_1u"]:
        med = per_fix[col].median()
        nul = (per_fix[col] == 0).mean()
        print(f"  {col:>7}  mediaan {med:5.1f}   fixtures met nul: {nul:6.1%}")

    return c[["fixture_id", "lead_h", "lead_band", "close_captured_at",
              "close_home", "close_draw", "close_away", "kickoff_at"]]


# =====================================================================
# C + D — GEVOLG VOOR DE GEMETEN CLV
# =====================================================================

def audit_clv(picks: pd.DataFrame, link: pd.DataFrame,
              proxy: pd.DataFrame, run_times: pd.DataFrame | None) -> pd.DataFrame:
    df = picks.merge(link, on="match_id", how="inner")
    df = df.merge(proxy, on="fixture_id", how="inner")
    if df.empty:
        print("\nGeen picks te koppelen aan een closing; C en D overgeslagen.")
        return df

    sel = df["selection"]
    df["odds_genomen"] = np.select(
        [sel == "HOME", sel == "DRAW", sel == "AWAY"],
        [df["odds_home"], df["odds_draw"], df["odds_away"]], default=np.nan)
    df["odds_close"] = np.select(
        [sel == "HOME", sel == "DRAW", sel == "AWAY"],
        [df["close_home"], df["close_draw"], df["close_away"]], default=np.nan)

    kop("C — is de closing-prijs uberhaupt een andere prijs?")
    gelijk = np.isclose(df["odds_genomen"], df["odds_close"], rtol=0, atol=1e-9)
    print(f"picks gekoppeld aan een closing: {len(df)}")
    print(f"  closing-odd exact gelijk aan genomen odd: {gelijk.mean():6.1%}"
          f"  ({int(gelijk.sum())} picks)")
    beweging = (df["odds_close"] - df["odds_genomen"]).abs()
    print(f"  mediane absolute prijsverandering: {beweging.median():.4f}")
    print("  (een hoog gelijk-aandeel betekent dat de CLV voor die picks")
    print("   mechanisch rond nul ligt en dus niets meet)")

    if run_times is not None and "run_id" in df.columns:
        df = df.merge(run_times, on="run_id", how="left")
        if "pick_created_at" in df.columns:
            venster = (df["close_captured_at"] - df["pick_created_at"]).dt.total_seconds() / 3600.0
            geldig = venster.notna()
            if geldig.any():
                print(f"\n  venster tussen pick-moment en closing (uren):")
                print(f"    mediaan {venster[geldig].median():.2f}   "
                      f"negatief of nul: {(venster[geldig] <= 0).mean():.1%}")
                print("    (negatief = de 'closing' is ouder dan de pick zelf)")

    # Devig de closing en bereken de kernmetriek van het rapport
    df = compute_market_probs(
        df, odds_cols=("close_home", "close_draw", "close_away"),
        out_cols=("p_close_home", "p_close_draw", "p_close_away"))
    df["p_close_sel"] = np.select(
        [sel == "HOME", sel == "DRAW", sel == "AWAY"],
        [df["p_close_home"], df["p_close_draw"], df["p_close_away"]], default=np.nan)
    df["edge_vs_close"] = df["p_close_sel"] * df["odds_genomen"] - 1.0

    kop("D — hangt de gemeten CLV af van de kwaliteit van de proxy?")
    g = df.groupby("lead_band", observed=True).agg(
        n=("edge_vs_close", "size"),
        clv=("edge_vs_close", "mean"),
        gem_lead_h=("lead_h", "mean"),
        gelijke_odds=("odds_genomen", "size"),
    )
    g["clv_pct"] = (g["clv"] * 100).round(2)
    g = g.reindex(LEAD_LABELS).dropna(subset=["n"])
    print(g[["n", "gem_lead_h", "clv_pct"]].round(2).to_string())

    schoon = df[df["lead_h"] <= 3]
    if len(schoon) >= 30:
        print(f"\nAlleen picks met een closing binnen 3 uur voor aftrap:")
        print(f"  n = {len(schoon)}   CLV = {schoon['edge_vs_close'].mean()*100:+.2f}%")
        print(f"  (vergelijk met alle picks: {df['edge_vs_close'].mean()*100:+.2f}%)")
    else:
        print(f"\nTe weinig picks met een closing binnen 3 uur ({len(schoon)}) "
              "voor een aparte schatting.")

    print("\nLezen: loopt de CLV op naarmate de closing VERDER van de aftrap")
    print("ligt, dan is de gemeten edge een artefact van een slechte proxy.")
    print("Is hij vlak, dan ligt de oorzaak van het bucket-patroon elders.")
    return df


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-csv", action="store_true")
    a = ap.parse_args()

    print("AUDIT — is de closing-proxy in clv_report.py een echte closing?")

    picks = load_picks()
    link, linknaam = load_link()
    picks_fix = picks.merge(link, on="match_id", how="inner")
    ids = picks_fix["fixture_id"].dropna().astype(int).unique().tolist()
    print(f"[load] fixtures achter de picks: {len(ids)}")

    snaps = load_snapshots(ids)
    kickoffs = load_kickoffs(ids)
    closing = build_closing(snaps, kickoffs)

    proxy = audit_proxy(snaps, kickoffs, closing)
    df = audit_clv(picks, link, proxy, load_run_times())

    if a.export_csv and not df.empty:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pad = EXPORT_DIR / f"audit_closing_line_{stamp}.csv"
        df.to_csv(pad, index=False)
        print(f"\nweggeschreven: {pad}")


if __name__ == "__main__":
    main()