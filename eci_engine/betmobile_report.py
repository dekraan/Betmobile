"""
betmobile_report.py

HET TOTAALRAPPORT. Eén run, altijd dezelfde secties, geen zoektocht.

ONTWERPPRINCIPE - waarom dit rapport zichzelf niet kan tegenspreken:
Elk optimistisch cijfer staat in DEZELFDE tabel naast zijn tegencheck.
ROI verschijnt nooit zonder aantal bets, betrouwbaarheidsinterval en CLV.
Zo kan "+15% ROI!" nooit meer los circuleren van "maar -6% vs de closing
line en n is te klein" - beide staan op één regel.

Dit is bewust GEEN ontdekkingsmachine. Het zoekt niet naar de beste
segmenten (dat doet research_backtest.py, met alle bijbehorende risico's
op toevalsvondsten). Dit rapport toont een vaste set diagnostiek, zodat
twee runs op verschillende dagen vergelijkbaar zijn.

SECTIES
  0. Register van beantwoorde vragen (wat is al dicht?)
  1. Staat van de data
  2. ECI-gezondheid (odds-vrij)
  3. Marktstructuur: Bet365 vs Pinnacle (benchmark)
  4. Pickprestatie, altijd gekoppeld aan CLV
  5. Benchmarks: doet selectie beter dan simpele alternatieven?
  6. Eindoordeel in gewone taal

Gebruik:
    python betmobile_report.py
    python betmobile_report.py --export-csv
    python betmobile_report.py --no-refresh
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import OUTPUT_DIR
from db import db_engine, refresh_source_views, relation_exists

from fit_calibration import (
    DEFAULT_SCHEMA,
    DEFAULT_SOURCE,
    load_match_frame,
    prepare_match_frame,
    print_header,
    print_table,
)
from shared_buckets import ODDS_BINS_REPORT, ODDS_LABELS_REPORT
from eci_quality import calibration_table, to_long, build_team_history, rating_lag_analysis, wilson_ci

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

BET365_ID = 8
PINNACLE_ID = 4
FREEZE_DATE = "2026-08-18"

# Vragen die volgens vooraf vastgelegd protocol zijn afgesloten.
# Staat bovenaan elk rapport, zodat een "interessante" bevinding hieronder
# nooit stilzwijgend een gesloten vraag heropent.
SETTLED_QUESTIONS = [
    ("Voegen ECI-kansen iets toe bovenop de markt?",
     "NEE", "blend-gewicht w=0 op 10.103 wedstrijden, train en test"),
    ("Idem, specifiek in de pick-zone?",
     "NEE", "w=0, ook op de testhelft (fit_calibration --pick-zone)"),
    ("Bevat ECI conditionele info die een blend mist?",
     "NEE", "stacking beta=-0.02, bootstrap-CI sluit nul in (stacking_test)"),
    ("Verslaan de picks de closing line (timing-edge)?",
     "NEE", "edge vs close -6%, t=-4.8 (clv_report)"),
    ("Rangschikt ECI wedstrijden correct (los van de markt)?",
     "JA", "monotonie 100%, wel te extreme kansen aan de uiteinden"),
    ("Is ECI te traag met ratingaanpassing?",
     "JA, licht", "momentum-slope t=+2.73; vorm-3 niet significant"),
    ("Bestaat er een beter koopmoment voor de aftrap?",
     "NEE, klein", "vroeg ~1,6pt beter maar andere wedstrijden per venster"),
]


# =====================================================================
# HULP
# =====================================================================

def q(sql: str, params: dict | None = None) -> pd.DataFrame:
    with db_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def object_exists(name: str, schema: str = "public") -> bool:
    """Bestaat de relatie? Werkt voor tabellen EN views.

    (db.relation_exists kijkt alleen naar views/matviews, waardoor gewone
    tabellen als 'ONTBREEKT' werden gerapporteerd.)
    """
    try:
        r = q("SELECT to_regclass(:full) IS NOT NULL AS ok", {"full": f"{schema}.{name}"})
        return bool(r.iloc[0]["ok"])
    except Exception:  # noqa: BLE001
        return False


def roi_row(label: str, profit: np.ndarray, extra: dict | None = None) -> dict:
    """ROI met n en 95%-interval - nooit los van elkaar te lezen."""
    n = len(profit)
    roi = float(profit.mean()) if n else np.nan
    se = float(profit.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    row = {
        "segment": label,
        "bets": n,
        "roi": roi,
        "roi_lo": roi - 1.96 * se if n > 1 else np.nan,
        "roi_hi": roi + 1.96 * se if n > 1 else np.nan,
    }
    row.update(extra or {})
    return row


def devig(odds: pd.DataFrame) -> pd.DataFrame:
    """
    Ge-devigde kansen volgens Shin - dezelfde methode als de rest van het
    systeem. Eerder stond hier een eigen proportionele berekening, waardoor
    sectie 3b en 4 met een andere methode rekenden dan sectie 3b-1 en de
    CLV-cijfers structureel te negatief uitvielen.
    """
    from prob_calibration import devig_shin

    arr = odds.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(arr).all(axis=1) & (arr > 1.01).all(axis=1)
    out = np.full(arr.shape, np.nan)
    if ok.any():
        p, _ = devig_shin(arr[ok])
        out[ok] = p
    return pd.DataFrame(out, index=odds.index, columns=odds.columns)


# =====================================================================
# 0. REGISTER
# =====================================================================

def section_settled() -> pd.DataFrame:
    print_header("0. REGISTER VAN BEANTWOORDE VRAGEN")
    print("Deze vragen zijn volgens vooraf vastgelegd protocol afgesloten.")
    print("Een bevinding verderop heropent ze niet zonder nieuw protocol.\n")
    df = pd.DataFrame(SETTLED_QUESTIONS, columns=["vraag", "antwoord", "bewijs"])
    print(df.to_string(index=False))
    return df


# =====================================================================
# 1. STAAT VAN DE DATA
# =====================================================================

def section_data_state() -> pd.DataFrame:
    print_header("1. STAAT VAN DE DATA")
    rows = []

    try:
        m = q("""
            SELECT COUNT(*) AS n,
                   MIN(COALESCE(oddspedia_date, eci_date)) AS van,
                   MAX(COALESCE(oddspedia_date, eci_date)) AS tot
            FROM public.eci_oddspedia_matches
        """).iloc[0]
        rows.append({"bron": "eci_oddspedia_matches (historie)", "rijen": int(m["n"]),
                     "van": str(m["van"])[:10], "tot": str(m["tot"])[:10]})
    except Exception as exc:  # noqa: BLE001
        rows.append({"bron": "eci_oddspedia_matches", "rijen": f"fout: {exc}"})

    try:
        s = q("""
            SELECT bookmaker_id, market_key,
                   COUNT(*) AS n, COUNT(DISTINCT fixture_id) AS fixtures,
                   MIN(captured_at) AS van, MAX(captured_at) AS tot
            FROM public.odds_values_snapshots
            GROUP BY bookmaker_id, market_key
            ORDER BY bookmaker_id, market_key
        """)
        names = {BET365_ID: "Bet365", PINNACLE_ID: "Pinnacle"}
        for _, r in s.iterrows():
            rows.append({
                "bron": f"snapshots {names.get(int(r['bookmaker_id']), r['bookmaker_id'])} / {r['market_key']}",
                "rijen": int(r["n"]), "fixtures": int(r["fixtures"]),
                "van": str(r["van"])[:10], "tot": str(r["tot"])[:10],
            })
    except Exception as exc:  # noqa: BLE001
        rows.append({"bron": "odds_values_snapshots", "rijen": f"fout: {exc}"})

    for tbl, label in [
        ("picks_evaluated_unique_v", "picks (uniek)"),
        ("picks_evaluated", "picks (ruwe rijen)"),
    ]:
        try:
            if not object_exists(tbl):
                rows.append({"bron": label, "rijen": "ONTBREEKT"})
                continue
            p = q(f"""
                SELECT COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS settled
                FROM public.{tbl}
            """).iloc[0]
            rows.append({"bron": label, "rijen": int(p["n"]), "settled": int(p["settled"])})
        except Exception as exc:  # noqa: BLE001
            rows.append({"bron": label, "rijen": f"fout: {exc}"})

    df = pd.DataFrame(rows)
    print(df.fillna("").to_string(index=False))

    # Versheid: draait de pipeline nog?
    try:
        last = q("SELECT MAX(captured_at) AS t FROM public.odds_values_snapshots").iloc[0]["t"]
        if pd.notna(last):
            age = (datetime.now(timezone.utc) - pd.Timestamp(last).tz_convert("UTC")).total_seconds() / 3600
            status = "OK" if age < 24 else "LET OP: pipeline lijkt stil te liggen"
            print(f"\nlaatste snapshot: {age:.1f} uur geleden -> {status}")
    except Exception:  # noqa: BLE001
        pass

    return df


# =====================================================================
# 2. ECI-GEZONDHEID
# =====================================================================

def section_eci_health(df: pd.DataFrame) -> dict:
    print_header("2. ECI-GEZONDHEID (ODDS-VRIJ)")
    long = to_long(df)
    cal = calibration_table(long)

    print("2a. Kalibratie: claimt ECI wat het levert?")
    print(cal[["prob_bucket", "n", "claimed", "actual", "gap", "verdict"]].round(4).to_string(index=False))

    mono = cal[cal["n"] >= 30]
    inc = mono["actual"].diff().dropna()
    share_up = float((inc > 0).mean()) if len(inc) else np.nan
    corr = float(np.corrcoef(long["p_eci"], long["hit"])[0, 1])

    flagged = cal[cal["verdict"] == "TE ZELFVERZEKERD"]["prob_bucket"].tolist()
    print(
        f"\nmonotonie: {share_up:.0%} van de buckets stijgt | correlatie {corr:.3f}\n"
        f"te zelfverzekerd in: {flagged if flagged else 'geen bucket'}"
    )

    hist = build_team_history(df)
    _, _, stats = rating_lag_analysis(df, hist)
    for name, s in stats.items():
        verdict = "restinformatie aanwezig" if abs(s["t"]) > 2 else "geen restinformatie"
        print(f"2b. {name}: beta={s['beta']:+.6f} t={s['t']:+.2f} -> {verdict}")

    return {"calibration": cal, "monotonic": share_up, "lag": stats}


# =====================================================================
# 3. MARKTSTRUCTUUR: BET365 VS PINNACLE
# =====================================================================

def section_market_comparison() -> dict:
    print_header("3. MARKTSTRUCTUUR: BET365 VS PINNACLE")
    print(
        "Pinnacle geldt als scherpste markt en dient hier als benchmark\n"
        "(niet als speelbare bookmaker). Twee vragen: is Bet365 duurder,\n"
        "en loopt Bet365 achter op Pinnacle?"
    )

    snaps = q("""
        WITH s AS (
            SELECT fixture_id, bookmaker_id, captured_at,
                   MAX(odd) FILTER (WHERE label = 'Home') AS h,
                   MAX(odd) FILTER (WHERE label = 'Draw') AS d,
                   MAX(odd) FILTER (WHERE label = 'Away') AS a
            FROM public.odds_values_snapshots
            WHERE market_key = '1x2' AND bookmaker_id IN (:b, :p)
            GROUP BY fixture_id, bookmaker_id, captured_at
        )
        SELECT * FROM s WHERE h IS NOT NULL AND d IS NOT NULL AND a IS NOT NULL
    """, {"b": BET365_ID, "p": PINNACLE_ID})

    if snaps.empty:
        print("Geen 1x2-snapshots gevonden.")
        return {}

    snaps["captured_at"] = pd.to_datetime(snaps["captured_at"], utc=True)
    for c in ["h", "d", "a"]:
        snaps[c] = pd.to_numeric(snaps[c], errors="coerce")
    snaps = snaps.dropna(subset=["h", "d", "a"])
    snaps["overround"] = (1 / snaps["h"] + 1 / snaps["d"] + 1 / snaps["a"])

    # ---- 3a. Marge per bookmaker ----
    marge = (
        snaps.groupby("bookmaker_id")
        .agg(snapshots=("overround", "size"), fixtures=("fixture_id", "nunique"),
             marge=("overround", lambda s: s.mean() - 1))
        .reset_index()
    )
    marge["bookmaker"] = marge["bookmaker_id"].map({BET365_ID: "Bet365", PINNACLE_ID: "Pinnacle"})
    print_table("3a. MARGE (lager = scherper)", marge[["bookmaker", "fixtures", "snapshots", "marge"]])

    have_both = set(snaps[snaps["bookmaker_id"] == BET365_ID]["fixture_id"]) & set(
        snaps[snaps["bookmaker_id"] == PINNACLE_ID]["fixture_id"]
    )
    print(f"\nfixtures met beide bookmakers: {len(have_both)}")
    if len(have_both) < 30:
        print("Te weinig overlap voor een prijsvergelijking.")
        return {"margins": marge}

    # ---- 3b. Prijsvergelijking op vergelijkbare momenten ----
    # Per fixture de laatste snapshot van elk; alleen als ze qua tijd dicht
    # bij elkaar liggen (< 6 uur), anders vergelijk je appels met peren.
    last = (
        snaps.sort_values("captured_at")
        .groupby(["fixture_id", "bookmaker_id"], as_index=False)
        .last()
    )
    b = last[last["bookmaker_id"] == BET365_ID].set_index("fixture_id")
    p = last[last["bookmaker_id"] == PINNACLE_ID].set_index("fixture_id")
    common = b.index.intersection(p.index)
    b, p = b.loc[common], p.loc[common]

    hours_apart = (b["captured_at"] - p["captured_at"]).abs().dt.total_seconds() / 3600
    ok = hours_apart < 6
    b, p = b[ok], p[ok]
    print(f"vergelijkbare snapshotparen (<6u uit elkaar): {len(b)}")
    if len(b) < 30:
        print("Te weinig vergelijkbare paren.")
        return {"margins": marge}

    p_pin = devig(p[["h", "d", "a"]])
    rows = []
    for lbl, col in [("thuis", "h"), ("gelijk", "d"), ("uit", "a")]:
        edge = p_pin[col].to_numpy() * b[col].to_numpy() - 1.0
        rows.append({
            "uitkomst": lbl,
            "n": len(edge),
            "gem_edge_vs_pinnacle": float(np.mean(edge)),
            "aandeel_edge_positief": float(np.mean(edge > 0)),
            "beste_1pct": float(np.percentile(edge, 99)),
        })
    prijs = pd.DataFrame(rows)
    print_table("3b. BET365-PRIJS GEMETEN TEGEN PINNACLE-KANS", prijs)
    print(
        "gem_edge < 0 is normaal en gelijk aan Bet365' marge. Interessant is\n"
        "alleen of er een STAART is: hoe vaak biedt Bet365 een prijs die zelfs\n"
        "tegen Pinnacle's kans nog positief uitpakt (aandeel_edge_positief)."
    )

    return {"margins": marge, "prices": prijs, "pairs": len(b)}


# =====================================================================
# 3b. TIMING: WANNEER KOPEN?
# =====================================================================

def section_timing() -> dict:
    """
    Op welk moment voor de aftrap kreeg je de beste prijs?

    Draait timing_report door en toont alleen de kern. Uitkomst tot nu toe:
    de prijs verandert gemiddeld nauwelijks, maar vroeg kopen kost in elk
    geval niets en scheelt mogelijk ~1,5 punt.
    """
    print_header("3b. TIMING: WANNEER KOPEN?")
    try:
        from timing_report import (
            load_picks as _lp, load_link as _ll, load_kickoffs as _lk,
            load_snapshots as _ls, build_closing as _bc,
            price_at_windows, build_timing_frame, summarize_windows,
            summarize_by_class,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"timing_report niet beschikbaar: {exc}")
        return {}

    try:
        picks = _lp()
        if picks.empty:
            print("Geen gesettelde picks.")
            return {}
        link, _ = _ll()
        matched = picks.merge(link, on="match_id", how="inner")
        fids = sorted(set(int(x) for x in matched["fixture_id"].dropna()))
        if not fids:
            print("Geen picks gekoppeld aan een fixture.")
            return {}

        kickoffs = _lk(fids)
        snaps = _ls(fids)
        closing = _bc(snaps, kickoffs)
        windows = price_at_windows(snaps, kickoffs)
        if windows.empty:
            print("Geen prijsverloop beschikbaar.")
            return {}

        df = build_timing_frame(picks, link, windows, closing, dedupe=True)
        if df.empty:
            print("Geen picks met bruikbaar prijsverloop.")
            return {}

        summary = summarize_windows(df)
        print_table("3b-1. PRIJS EN CLV PER KOOPMOMENT", summary)
        print(
            "aandeel_beter_dan_close daalt richting de aftrap omdat je dan\n"
            "steeds meer DE closing line zelf koopt - dat is meetkunde, geen edge.\n"
            "De echte vraag is of het vroegste venster ruim boven 50% uitkomt."
        )

        per_class = summarize_by_class(df)
        if not per_class.empty:
            print_table("3b-2. VROEG VS LAAT PER COMPETITIEKLASSE", per_class)

        return {"timing": summary, "timing_by_class": per_class}
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] timing-sectie overgeslagen: {exc}")
        return {}


# =====================================================================
# 4. PICKPRESTATIE, GEKOPPELD AAN CLV
# =====================================================================

def load_picks_with_clv() -> pd.DataFrame:
    """Picks + closing line van Bet365 en (indien beschikbaar) Pinnacle."""
    src = "picks_evaluated_unique_v"
    if not object_exists(src):
        src = "picks_evaluated"

    picks = q(f"""
        SELECT match_id, competition, date, date_ts, selection, outcome,
               odds_home, odds_draw, odds_away, pick_type, pick_tier,
               rule_strength_adj
        FROM public.{src}
        WHERE outcome IN ('WIN','LOSS') AND selection IN ('HOME','DRAW','AWAY')
    """)
    if picks.empty:
        return picks

    link = q("SELECT match_id, fixture_id FROM public.eci_fixture_link_mv").dropna()
    picks = picks.merge(link, on="match_id", how="left")

    close = q("""
        WITH s AS (
            SELECT s.fixture_id, s.bookmaker_id, s.captured_at,
                   MAX(s.odd) FILTER (WHERE s.label='Home') AS h,
                   MAX(s.odd) FILTER (WHERE s.label='Draw') AS d,
                   MAX(s.odd) FILTER (WHERE s.label='Away') AS a
            FROM public.odds_values_snapshots s
            JOIN public.fixtures f ON f.fixture_id = s.fixture_id
            WHERE s.market_key='1x2' AND s.captured_at <= f.date_utc
            GROUP BY s.fixture_id, s.bookmaker_id, s.captured_at
        )
        SELECT DISTINCT ON (fixture_id, bookmaker_id)
               fixture_id, bookmaker_id, h, d, a
        FROM s
        WHERE h IS NOT NULL AND d IS NOT NULL AND a IS NOT NULL
        ORDER BY fixture_id, bookmaker_id, captured_at DESC
    """)

    for bid, tag in [(BET365_ID, "b365"), (PINNACLE_ID, "pin")]:
        part = close[close["bookmaker_id"] == bid][["fixture_id", "h", "d", "a"]]
        part = part.rename(columns={c: f"close_{tag}_{c}" for c in ["h", "d", "a"]})
        picks = picks.merge(part, on="fixture_id", how="left")

    sel = picks["selection"]
    picks["odds_taken"] = np.select(
        [sel == "HOME", sel == "DRAW", sel == "AWAY"],
        [picks["odds_home"], picks["odds_draw"], picks["odds_away"]], default=np.nan)
    picks["profit"] = np.where(picks["outcome"] == "WIN", picks["odds_taken"] - 1.0, -1.0)

    for tag in ["b365", "pin"]:
        cols = [f"close_{tag}_h", f"close_{tag}_d", f"close_{tag}_a"]
        if not set(cols) <= set(picks.columns):
            picks[f"edge_{tag}"] = np.nan
            continue
        sub = picks[cols].apply(pd.to_numeric, errors="coerce")
        valid = (sub > 1.01).all(axis=1)
        dv = devig(sub.where(valid))
        p_sel = np.select(
            [sel == "HOME", sel == "DRAW", sel == "AWAY"],
            [dv[cols[0]], dv[cols[1]], dv[cols[2]]], default=np.nan)
        picks[f"edge_{tag}"] = p_sel * picks["odds_taken"] - 1.0

    picks["date_dt"] = pd.to_datetime(
        picks["date_ts"].fillna(picks["date"].astype(str)), errors="coerce", utc=True)
    return picks


def summarize_picks(picks: pd.DataFrame, by: str | None, label: str) -> pd.DataFrame:
    """ROI en CLV altijd samen - dat is het hele punt van dit rapport."""
    groups = [(label, picks)] if by is None else list(picks.groupby(by, observed=True))
    rows = []
    for name, part in groups:
        if part.empty:
            continue
        row = roi_row(str(name), part["profit"].to_numpy(float))
        for tag, col in [("clv_b365", "edge_b365"), ("clv_pin", "edge_pin")]:
            vals = part[col].dropna().to_numpy(float)
            row[f"{tag}_n"] = len(vals)
            row[tag] = float(vals.mean()) if len(vals) else np.nan
            if len(vals) > 1:
                se = vals.std(ddof=1) / np.sqrt(len(vals))
                row[f"{tag}_t"] = float(vals.mean() / se) if se else np.nan
            else:
                row[f"{tag}_t"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def section_picks(picks: pd.DataFrame) -> dict:
    print_header("4. PICKPRESTATIE (ROI ALTIJD NAAST CLV)")
    if picks.empty:
        print("Geen gesettelde picks.")
        return {}

    print(
        "Lees elke regel in zijn geheel: roi zonder [roi_lo, roi_hi] en zonder\n"
        "clv is betekenisloos. clv_* < 0 betekent dat de markt de genomen prijs\n"
        "achteraf slecht vond, ongeacht of de bet won."
    )

    freeze = pd.Timestamp(FREEZE_DATE, tz="UTC")
    parts = [
        summarize_picks(picks, None, "ALLES"),
        summarize_picks(picks[picks["date_dt"] < freeze], None, f"voor freeze (<{FREEZE_DATE})"),
        summarize_picks(picks[picks["date_dt"] >= freeze], None, f"SINDS FREEZE (>={FREEZE_DATE})"),
    ]
    total = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    print_table("4a. TOTAAL EN PERIODE", total)

    if "pick_tier" in picks.columns and picks["pick_tier"].notna().any():
        print_table("4b. PER TIER", summarize_picks(picks[picks["pick_tier"].notna()], "pick_tier", "tier"))
    if "pick_type" in picks.columns:
        print_table("4c. PER PICK TYPE", summarize_picks(picks, "pick_type", "type"))

    picks = picks.copy()
    picks["odds_bucket"] = pd.cut(
        picks["odds_taken"], bins=ODDS_BINS_REPORT, labels=ODDS_LABELS_REPORT)
    print_table("4d. PER ODDS BUCKET", summarize_picks(picks, "odds_bucket", "odds"))

    return {"total": total}


# =====================================================================
# 5. BENCHMARKS
# =====================================================================

def _diff_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Verschil in gemiddelde ROI tussen twee onafhankelijke groepen."""
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan, np.nan)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (float(diff), float(diff - 1.96 * se), float(diff + 1.96 * se))


def section_benchmarks(df: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    print_header("5. BENCHMARKS: DOET SELECTIE BETER DAN SIMPELE ALTERNATIEVEN?")
    print(
        "De eerlijke vergelijking voor 'ECI helpt kansrijke wedstrijden kiezen'\n"
        "is niet nul, maar: wat levert blind op favorieten inzetten op?"
    )

    mkt = df[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(float)
    eci = df[["mdl_home", "mdl_draw", "mdl_away"]].to_numpy(float)
    odds_mat = df[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    y = df["y_idx"].to_numpy(int)
    idx = np.arange(len(df))

    fav_idx, eci_idx = mkt.argmax(axis=1), eci.argmax(axis=1)
    mkt_max, eci_max = mkt.max(axis=1), eci.max(axis=1)
    fav_profit = np.where(fav_idx == y, odds_mat[idx, fav_idx] - 1.0, -1.0)
    eci_profit = np.where(eci_idx == y, odds_mat[idx, eci_idx] - 1.0, -1.0)
    agree = eci_idx == fav_idx

    rows = [
        roi_row("blind op marktfavoriet (alle wedstrijden)", fav_profit),
        roi_row("blind op ECI-favoriet", eci_profit),
        roi_row("ECI en markt zijn het EENS", fav_profit[agree]),
        roi_row("ECI en markt zijn het ONEENS", eci_profit[~agree]),
    ]

    bench = pd.DataFrame(rows)
    print(bench.round(4).to_string(index=False))

    # ---------------------------------------------------------------
    # DISAMBIGUATIE
    # "eens EN ECI >= 60%" selecteert vooral korte odds, en favorieten
    # verliezen sowieso minder dan longshots (favourite-longshot bias).
    # Vraag: doet ECI het werk, of doet de marktkans dat in zijn eentje?
    # ---------------------------------------------------------------
    print_header("5b. DISAMBIGUATIE: IS HET ECI, OF IS HET DE MARKTKANS?")
    print(
        "'eens EN ECI>=60%' selecteert korte odds. Favorieten verliezen sowieso\n"
        "minder. Daarom hier de eerlijke tegenhanger: dezelfde selectie op\n"
        "MARKTKANS alleen, zonder ECI."
    )

    eci_filter = agree & (eci_max >= 0.60)
    mkt_filter = mkt_max >= 0.60

    dis = pd.DataFrame([
        roi_row("A. eens EN ECI >= 60% (jouw filter)", fav_profit[eci_filter],
                {"gem_odds": float(odds_mat[idx, fav_idx][eci_filter].mean())}),
        roi_row("B. markt >= 60% (ZONDER ECI)", fav_profit[mkt_filter],
                {"gem_odds": float(odds_mat[idx, fav_idx][mkt_filter].mean())}),
    ])
    print(dis.round(4).to_string(index=False))

    d, lo, hi = _diff_test(fav_profit[eci_filter], fav_profit[mkt_filter])
    print(
        f"\nverschil A - B: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]\n"
        "Sluit dit interval nul in, dan voegt ECI niets toe boven 'kies\n"
        "wedstrijden waar de markt de favoriet hoog inschat'."
    )

    # De strengste vorm: BINNEN de markt>=60%-groep, maakt ECI verschil?
    # Nu is de marktkans gelijkgetrokken, dus de bias kan het niet verklaren.
    print_header("5c. STRENGSTE TEST: BINNEN MARKT >= 60%, MAAKT ECI VERSCHIL?")
    within_yes = mkt_filter & agree & (eci_max >= 0.60)
    within_no = mkt_filter & ~(agree & (eci_max >= 0.60))
    within = pd.DataFrame([
        roi_row("markt >= 60% MET ECI-bevestiging", fav_profit[within_yes],
                {"gem_mkt_kans": float(mkt_max[within_yes].mean())}),
        roi_row("markt >= 60% ZONDER ECI-bevestiging", fav_profit[within_no],
                {"gem_mkt_kans": float(mkt_max[within_no].mean())}),
    ])
    # Hitrate-residu: won de favoriet vaker dan de markt claimde? Deze maat
    # is veel minder ruizig dan ROI (geen odds-variantie erin) en heeft
    # daardoor veel meer onderscheidend vermogen bij deze aantallen.
    fav_hit = (fav_idx == y).astype(float)
    resid = fav_hit - mkt_max
    within["hitrate"] = [float(fav_hit[within_yes].mean()), float(fav_hit[within_no].mean())]
    within["resid_vs_markt"] = [float(resid[within_yes].mean()), float(resid[within_no].mean())]
    print(within.round(4).to_string(index=False))

    d2, lo2, hi2 = _diff_test(fav_profit[within_yes], fav_profit[within_no])
    d3, lo3, hi3 = _diff_test(resid[within_yes], resid[within_no])
    verdict = (
        "ECI VOEGT IETS TOE binnen dezelfde marktkans"
        if not (np.isnan(lo3) or (lo3 <= 0 <= hi3))
        else "geen aantoonbaar effect van ECI-bevestiging"
    )
    print(
        f"\nverschil in ROI      : {d2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]  (ruizig)\n"
        f"verschil in hitrate-residu: {d3:+.4f} [{lo3:+.4f}, {hi3:+.4f}]  (gevoeliger)\n"
        f"-> {verdict}\n"
        "Het residu is de doorslaggevende maat: ROI bevat ook odds-variantie\n"
        "en heeft daardoor bij deze aantallen te weinig onderscheidend vermogen."
    )

    if not picks.empty:
        bench = pd.concat(
            [bench, pd.DataFrame([roi_row("onze daadwerkelijke picks", picks["profit"].to_numpy(float))])],
            ignore_index=True,
        )

    print(
        "\nAlle regels op dezelfde historische data (deels tuningperiode voor de\n"
        "laatste regel). Overlappen de intervallen, dan is het verschil niet\n"
        "aangetoond. Verwacht: alles rond -5% (de marge)."
    )
    return pd.concat([bench, dis, within], ignore_index=True)


# =====================================================================
# 6. EINDOORDEEL
# =====================================================================

def section_verdict(health: dict, market: dict, picks_res: dict) -> None:
    print_header("6. EINDOORDEEL")

    lines = []
    if health:
        mono = health.get("monotonic", np.nan)
        lines.append(
            f"ECI rangschikt {'correct' if mono and mono > 0.8 else 'niet consistent'} "
            f"({mono:.0%} van de buckets stijgt) maar overdrijft aan de uiteinden."
        )
    if market.get("prices") is not None:
        pr = market["prices"]
        share = pr["aandeel_edge_positief"].mean()
        lines.append(
            f"Bet365 biedt in {share:.0%} van de gevallen een prijs die tegen "
            "Pinnacle's kans positief uitpakt; dat is de enige plek waar nog "
            "onbenutte ruimte zou kunnen zitten."
        )
    if market.get("prices") is None and not lines:
        pass
    if picks_res.get("total") is not None and not picks_res["total"].empty:
        t = picks_res["total"]
        oos = t[t["segment"].str.startswith("SINDS FREEZE")]
        if not oos.empty and int(oos.iloc[0]["bets"]) > 0:
            r = oos.iloc[0]
            lines.append(
                f"Out-of-sample sinds freeze: {int(r['bets'])} bets, ROI {r['roi']:+.1%} "
                f"[{r['roi_lo']:+.1%}, {r['roi_hi']:+.1%}], CLV {r['clv_b365']:+.1%}. "
                "Oordeel pas bij 150-200 bets."
            )
        else:
            lines.append("Out-of-sample sinds freeze: nog geen gesettelde picks.")

    for line in lines:
        print(f"- {line}")

    print(
        "\nStaande afspraak: de vragen in sectie 0 zijn dicht. Een opvallend\n"
        "cijfer in dit rapport is een aanleiding voor een NIEUWE vooraf\n"
        "vastgelegde test, niet voor een conclusie."
    )


# =====================================================================
# MAIN
# =====================================================================

def run(source: str, schema: str, refresh: bool, export_csv: bool) -> None:
    print_header(f"BETMOBILE TOTAALRAPPORT - {datetime.now():%Y-%m-%d %H:%M}")

    settled = section_settled()
    data_state = section_data_state()

    if refresh:
        try:
            refresh_source_views()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] refresh mislukt: {exc}")

    df = prepare_match_frame(load_match_frame(source, schema)).reset_index(drop=True)
    health = section_eci_health(df)

    try:
        market = section_market_comparison()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] marktvergelijking overgeslagen: {exc}")
        market = {}

    try:
        timing = section_timing()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] timing overgeslagen: {exc}")
        timing = {}

    try:
        picks = load_picks_with_clv()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] picks laden mislukt: {exc}")
        picks = pd.DataFrame()

    picks_res = section_picks(picks) if not picks.empty else {}
    bench = section_benchmarks(df, picks)
    section_verdict(health, market, picks_res)

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tables = {
            "report_settled": settled,
            "report_data_state": data_state,
            "report_eci_calibration": health.get("calibration"),
            "report_market_margins": market.get("margins"),
            "report_market_prices": market.get("prices"),
            "report_picks": picks_res.get("total"),
            "report_benchmarks": bench,
            "report_timing": timing.get("timing"),
            "report_timing_by_class": timing.get("timing_by_class"),
        }
        for name, table in tables.items():
            if table is not None and not table.empty:
                path = EXPORT_DIR / f"{name}_{stamp}.csv"
                table.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"[export] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Betmobile totaalrapport")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()
    run(args.source, args.schema, not args.no_refresh, args.export_csv)


if __name__ == "__main__":
    main()