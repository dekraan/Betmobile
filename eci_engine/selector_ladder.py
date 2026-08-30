"""
selector_ladder.py

TEST 5 - Voegt de ECI-SELECTIE iets toe bovenop de markt?

Vier eerdere tests keken naar ECI als KANSSCHATTER (blend, stacking,
calibratie, CLV van de picks). Die zeiden allemaal nee. Deze test kijkt
naar iets anders: bepaalt ECI WELKE wedstrijden je speelt op een manier
die geld oplevert? Dat is een aparte vraag, want een selector kan werken
zonder dat de onderliggende kans goed is.

PREREGISTRATIE (vastgelegd voor de eerste run)
  Beoordeeld wordt de VORM van de ladder, niet een enkele drempel.
  Positief bewijs vereist BEIDE:
    (1) monotone stijging over minstens drie opeenvolgende drempels, en
    (2) de strengste drempel met CI die nul uitsluit.
  Rijen met n < 150 tellen niet mee in het oordeel.
  Verwachting op basis van test 1-4: vlak rond nul.

DRIE MATEN, BEWUST GESCHEIDEN
  roi           gerealiseerd rendement. Ruw, dus gevoelig voor de
                favourite-longshot bias: strengere filters duwen de odds
                omlaag en dat alleen al verandert de ROI.
  excess_roi    roi minus het gemiddelde van dezelfde prijsklasse. Corrigeert
                voor die bias. LET OP: de baseline komt uit dezelfde
                uitkomsten die getoetst worden, dus dit is een in-sample
                genormaliseerde maat. Beschrijvend bruikbaar, niet als
                formele toets.
  edge_shin     mkt_prob(gekozen kant) x odds - 1, met Shin-devig. Gebruikt
                GEEN enkele uitslag, dus geen in-sample probleem. Meet
                verwachte in plaats van gerealiseerde afwijking. Is per
                constructie negatief zolang je tegen de marge in speelt;
                wat telt is of hij OPLOOPT met de drempel.
  clv           p_close(gekozen kant) x odds_genomen - 1. Ex-post
                marktbeweging. Convergeert sneller dan roi.

DRIE LADDERS
  A  ECI-kans          is een sterker ECI-signaal winstgevender?
  B  rating gap        idem, maar op de ruwe rating in plaats van de kans.
  C  afwijking         eci_prob - mkt_prob op de gekozen kant. Dit is de
                       eigenlijke vraag van het project: heeft ECI iets dat
                       NIET al in de prijs zit? Ladder A kan positief zijn
                       terwijl ECI simpelweg de markt napraat; ladder C niet.

Gebruik:
    python selector_ladder.py
    python selector_ladder.py --export-csv
    python selector_ladder.py --no-clv        # sneller, slaat snapshots over
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from db import db_engine
from prob_calibration import compute_market_probs
from clv_report import (load_link, load_kickoffs, load_snapshots,
                        build_closing, load_run_times)

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Prijsklassen voor de excess_roi-baseline. Bewust grof: fijnere klassen
# geven te kleine cellen en dan corrigeer je vooral op ruis.
PRICE_BINS = [1.0, 1.30, 1.50, 1.75, 2.00, 2.50, 3.50, np.inf]
PRICE_LABELS = ["<1.30", "1.30-1.50", "1.50-1.75", "1.75-2.00",
                "2.00-2.50", "2.50-3.50", "3.50+"]

MIN_N_FOR_VERDICT = 150


# =====================================================================
# DATA
# =====================================================================

def load_universe() -> pd.DataFrame:
    """Alle wedstrijden die het model heeft gezien, met uitslag.

    Per match_id de LAATSTE run: dat is de toestand waarin het model zijn
    definitieve oordeel had. De eerste run is ongeschikt, want dan is
    min_snapshots nog niet gehaald en staat is_pick vrijwel altijd op false.
    """
    q = """
        WITH laatst AS (
            SELECT DISTINCT ON (match_id)
                   match_id, run_id, date, competition,
                   odds_home, odds_draw, odds_away,
                   prob_home, prob_draw, prob_away,
                   rating_gap, rating_home_edge,
                   is_pick, pick_tier, n_snapshots
            FROM public.model_match_snapshots
            ORDER BY match_id, run_id DESC
        ),
        ooit AS (
            SELECT match_id, BOOL_OR(is_pick) AS ooit_pick
            FROM public.model_match_snapshots
            GROUP BY match_id
        ),
        eerste AS (
            -- Vroegste prijs die het model zag. Nodig voor CLV: met de prijs
            -- uit de LAATSTE run is er geen venster meer tot de close en meet
            -- je de marge in plaats van de marktbeweging.
            SELECT DISTINCT ON (match_id)
                   match_id,
                   run_id     AS eerste_run,
                   odds_home  AS open_home,
                   odds_draw  AS open_draw,
                   odds_away  AS open_away
            FROM public.model_match_snapshots
            ORDER BY match_id, run_id
        ),
        res AS (
            SELECT DISTINCT ON (match_id) match_id,
                   split_part(eci_score, '-', 1)::int AS hg,
                   split_part(eci_score, '-', 2)::int AS ag
            FROM public.eci_data
            WHERE eci_score ~ '^[0-9]+-[0-9]+$'
              AND COALESCE(is_excluded, false) = false
            ORDER BY match_id
        )
        SELECT l.*, o.ooit_pick, r.hg, r.ag,
               e.eerste_run, e.open_home, e.open_draw, e.open_away
        FROM laatst l
        JOIN ooit   o USING (match_id)
        JOIN eerste e USING (match_id)
        JOIN res    r USING (match_id)
        WHERE l.odds_home IS NOT NULL
          AND l.odds_draw IS NOT NULL
          AND l.odds_away IS NOT NULL
          AND l.prob_home IS NOT NULL
          AND l.prob_away IS NOT NULL
          AND l.rating_gap IS NOT NULL
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    print(f"[load] universum: {len(df)} wedstrijden met odds, ECI en uitslag")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Kant, uitslag, marktkansen, afwijking en de drie maten."""
    df = df.copy()

    # Uitslag
    df["uitslag"] = np.select(
        [df["hg"] > df["ag"], df["hg"] < df["ag"]],
        ["HOME", "AWAY"],
        default="DRAW",
    )

    # ECI kiest de kant met zijn hoogste eigen kans (draw doet niet mee:
    # het systeem speelt geen draws).
    home_side = df["prob_home"] >= df["prob_away"]
    df["eci_side"] = np.where(home_side, "HOME", "AWAY")
    df["eci_prob"] = np.where(home_side, df["prob_home"], df["prob_away"])
    df["odds_taken"] = np.where(home_side, df["odds_home"], df["odds_away"])

    # Vroegste prijs voor dezelfde kant. ROI en edge_shin blijven op
    # odds_taken (het definitieve oordeel); CLV draait op odds_open, want
    # daar heb je wel een venster tot de close.
    df["odds_open"] = np.where(home_side, df["open_home"], df["open_away"])

    # Markt kiest de kant met de laagste odds
    df["mkt_side"] = np.where(df["odds_home"] <= df["odds_away"], "HOME", "AWAY")
    df["eens_met_markt"] = (df["eci_side"] == df["mkt_side"]).astype(float)

    # Ge-devigde marktkansen (Shin), zelfde implementatie als tier_assign
    df = compute_market_probs(df, out_cols=("mkt_home", "mkt_draw", "mkt_away"))
    df["mkt_prob"] = np.where(home_side, df["mkt_home"], df["mkt_away"])

    # De kernvariabele voor ladder C
    df["afwijking"] = df["eci_prob"] - df["mkt_prob"]

    # --- maat 1: gerealiseerde ROI ---
    won = df["uitslag"] == df["eci_side"]
    df["win"] = won.astype(float)
    df["profit"] = np.where(won, df["odds_taken"] - 1.0, -1.0)

    # --- maat 2: excess t.o.v. prijsklasse (in-sample baseline) ---
    df["prijsklasse"] = pd.cut(df["odds_taken"], bins=PRICE_BINS,
                               labels=PRICE_LABELS, right=False)
    basis = df.groupby("prijsklasse", observed=True)["profit"].transform("mean")
    df["excess"] = df["profit"] - basis

    # --- maat 3: verwachte edge volgens de markt (geen uitslagen) ---
    df["edge_shin"] = df["mkt_prob"] * df["odds_taken"] - 1.0

    return df


def add_clv(df: pd.DataFrame) -> pd.DataFrame:
    """CLV = p_close(gekozen kant) x odds_OPEN - 1.

    Gecorrigeerd na de eerste run. Daar draaide CLV op de prijs uit de
    LAATSTE modelrun, en die ligt vlak voor de aftrap. Gevolg: clv liep
    overal vrijwel gelijk op met edge_shin (beide rond -0,06) en slechts
    4,5% van de wedstrijden had positieve CLV, tegen 41% in clv_report.py.
    Er werd dus de marge gemeten in plaats van de marktbeweging.

    Nu: selectie op de laatste run (definitief oordeel), prijs uit de eerste
    run (vroegst geziene prijs). Dat is de eerlijke vraag - kocht je vroeg
    een prijs die daarna de goede kant op bewoog?

    clv_uren = venster tussen de eerste run en de close. Is dat klein, dan
    kan CLV per definitie weinig laten zien; die kolom hoort erbij gelezen
    te worden.
    """
    df = df.copy()
    df["clv"] = np.nan

    try:
        link, link_name = load_link()
    except Exception as exc:  # noqa: BLE001
        print(f"[clv] geen linkview beschikbaar ({exc}); CLV overgeslagen.")
        return df

    merged = df[["match_id"]].merge(link, on="match_id", how="inner")
    fixture_ids = merged["fixture_id"].dropna().astype(int).unique().tolist()
    if not fixture_ids:
        print("[clv] geen gekoppelde fixtures; CLV overgeslagen.")
        return df

    kickoffs = load_kickoffs(fixture_ids)
    snaps = load_snapshots(fixture_ids)
    closing = build_closing(snaps, kickoffs)
    if closing.empty:
        print("[clv] geen closing-snapshots gevonden; CLV overgeslagen.")
        return df

    closing = compute_market_probs(
        closing,
        odds_cols=("close_home", "close_draw", "close_away"),
        out_cols=("p_close_home", "p_close_draw", "p_close_away"),
    )

    lk = link.set_index("match_id")["fixture_id"]
    df["fixture_id"] = df["match_id"].map(lk)
    df = df.merge(
        closing[["fixture_id", "p_close_home", "p_close_away", "close_captured_at"]],
        on="fixture_id", how="left",
    )

    p_close_sel = np.where(df["eci_side"] == "HOME",
                           df["p_close_home"], df["p_close_away"])

    # CLV op de VROEGSTE prijs: alleen dan is er een venster tot de close.
    df["clv"] = p_close_sel * df["odds_open"] - 1.0

    # Controlemaat op de oude manier, zodat je in de output kunt zien dat het
    # verschil echt uit het koopmoment komt en niet uit de selectie.
    df["clv_laat"] = p_close_sel * df["odds_taken"] - 1.0

    # Hoe groot was dat venster? Zonder picks_run kunnen we het niet bepalen;
    # dan blijft de kolom leeg en weet je dat je hem niet kunt gebruiken.
    df["clv_uren"] = np.nan
    try:
        run_times = load_run_times()
    except Exception:  # noqa: BLE001
        run_times = None
    if run_times is not None and "eerste_run" in df.columns:
        rt = run_times.rename(columns={"run_id": "eerste_run"})
        df = df.merge(rt, on="eerste_run", how="left")
        t0 = pd.to_datetime(df["pick_created_at"], utc=True, errors="coerce")
        t1 = pd.to_datetime(df["close_captured_at"], utc=True, errors="coerce")
        df["clv_uren"] = (t1 - t0).dt.total_seconds() / 3600.0
    else:
        print("[clv] picks_run niet beschikbaar; venster (clv_uren) onbekend.")

    n = int(df["clv"].notna().sum())
    print(f"[clv] berekend voor {n} van {len(df)} wedstrijden "
          f"({n / max(len(df), 1):.0%} dekking)")
    if df["clv_uren"].notna().any():
        print(f"[clv] venster eerste run -> close: mediaan "
              f"{df['clv_uren'].median():.1f} uur, "
              f"{(df['clv_uren'] < 6).mean():.0%} onder 6 uur")
    print(f"[clv] gemiddeld op open-odds {df['clv'].mean():+.4f} "
          f"vs op laatste odds {df['clv_laat'].mean():+.4f}")
    return df


# =====================================================================
# LADDERS
# =====================================================================

def _ci95(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s))


def summarise(sub: pd.DataFrame, label: str) -> dict:
    return {
        "drempel": label,
        "n": len(sub),
        "gem_odds": sub["odds_taken"].mean(),
        "gem_open": sub["odds_open"].mean() if "odds_open" in sub else np.nan,
        "hitrate": sub["win"].mean(),
        "eens_markt": sub["eens_met_markt"].mean(),
        "roi": sub["profit"].mean(),
        "excess_roi": sub["excess"].mean(),
        "excess_ci": _ci95(sub["excess"]),
        "edge_shin": sub["edge_shin"].mean(),
        "n_clv": int(sub["clv"].notna().sum()) if "clv" in sub else 0,
        "clv": sub["clv"].mean() if "clv" in sub else np.nan,
        "clv_ci": _ci95(sub["clv"]) if "clv" in sub else np.nan,
        "clv_pos_pct": (sub["clv"] > 0).mean() if "clv" in sub else np.nan,
        "clv_laat": sub["clv_laat"].mean() if "clv_laat" in sub else np.nan,
        "clv_uren": sub["clv_uren"].median() if "clv_uren" in sub else np.nan,
    }


def build_ladder(df: pd.DataFrame, column: str, thresholds: list[float],
                 fmt: str = "{:.2f}") -> pd.DataFrame:
    rows = [summarise(df, "alles (baseline)")]
    for t in thresholds:
        sub = df[df[column] >= t]
        if sub.empty:
            continue
        rows.append(summarise(sub, f"{column} >= " + fmt.format(t)))
    return pd.DataFrame(rows)


def verdict(ladder: pd.DataFrame, metric: str, ci_col: str) -> str:
    """Past de preregistratie toe: monotonie over 3 stappen + CI sluit nul uit."""
    usable = ladder[(ladder["n"] >= MIN_N_FOR_VERDICT)
                    & (ladder["drempel"] != "alles (baseline)")]
    if len(usable) < 3:
        return f"ONBESLIST: minder dan 3 drempels met n >= {MIN_N_FOR_VERDICT}"

    vals = usable[metric].to_numpy()
    monotoon = any(
        vals[i] < vals[i + 1] < vals[i + 2]
        for i in range(len(vals) - 2)
    )
    laatste = usable.iloc[-1]
    ci_sluit_nul_uit = (
        pd.notna(laatste[ci_col])
        and abs(laatste[metric]) > laatste[ci_col]
        and laatste[metric] > 0
    )
    if monotoon and ci_sluit_nul_uit:
        return "POSITIEF: monotone stijging en strengste drempel significant"
    reden = []
    if not monotoon:
        reden.append("geen monotone stijging over 3 drempels")
    if not ci_sluit_nul_uit:
        reden.append("strengste drempel niet significant boven nul")
    return "NEGATIEF: " + " en ".join(reden)


def show(title: str, ladder: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    view = ladder.copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].round(4)
    print(view.to_string(index=False))
    print(f"  ROI-oordeel : {verdict(ladder, 'excess_roi', 'excess_ci')}")
    print(f"  CLV-oordeel : {verdict(ladder, 'clv', 'clv_ci')}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-csv", action="store_true")
    ap.add_argument("--no-clv", action="store_true")
    args = ap.parse_args()

    print("\n=== TEST 5: DOET DE ECI-SELECTIE IETS? ===")
    print("Preregistratie: de VORM van de ladder telt. Monotone stijging over")
    print("drie drempels EN een significante strengste drempel. Anders negatief.\n")

    df = prepare(load_universe())
    if not args.no_clv:
        df = add_clv(df)
    else:
        df["clv"] = np.nan
        df["clv_laat"] = np.nan
        df["clv_uren"] = np.nan

    print(f"\nAfwijking ECI - markt: gemiddeld {df['afwijking'].mean():+.4f}, "
          f"mediaan {df['afwijking'].median():+.4f}, "
          f"sd {df['afwijking'].std():.4f}")
    print(f"ECI eens met markt: {df['eens_met_markt'].mean():.1%} van alle wedstrijden")

    ladders = {
        "LADDER A - ECI-kans": build_ladder(
            df, "eci_prob", [0.50, 0.55, 0.58, 0.60, 0.65, 0.70]),
        "LADDER B - rating gap": build_ladder(
            df, "rating_gap", [200, 300, 500, 700, 1000], fmt="{:.0f}"),
        "LADDER C - afwijking ECI vs markt": build_ladder(
            df, "afwijking", [0.00, 0.02, 0.05, 0.08, 0.12], fmt="{:+.2f}"),
    }

    for title, lad in ladders.items():
        show(title, lad)

    # Ladder C ook naar beneden: wijkt ECI de ANDERE kant op af, dan is dat
    # historisch een waarschuwing (-12,4% ROI). Even controleren of dat klopt.
    neg = pd.DataFrame([
        summarise(df[df["afwijking"] <= t], f"afwijking <= {t:+.2f}")
        for t in (0.00, -0.02, -0.05)
        if not df[df["afwijking"] <= t].empty
    ])
    show("LADDER C-negatief - ECI ONDER de markt", neg)

    print("\nLET OP bij het lezen:")
    print("  excess_roi heeft een in-sample baseline (prijsklasse-gemiddelde uit")
    print("  dezelfde data). Beschrijvend bruikbaar, niet als formele toets.")
    print("  edge_shin gebruikt geen uitslagen en is dus wel schoon, maar meet")
    print("  verwachte in plaats van gerealiseerde afwijking.")

    if args.export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        for title, lad in ladders.items():
            naam = title.split(" - ")[0].replace(" ", "_").lower()
            path = EXPORT_DIR / f"selector_{naam}_{stamp}.csv"
            lad.to_csv(path, index=False)
            print(f"[export] {path}")


if __name__ == "__main__":
    main()