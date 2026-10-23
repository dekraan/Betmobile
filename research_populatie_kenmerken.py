"""
research_populatie_kenmerken.py
===============================

Vraag: welke cijfers voorspellen de uitslag BETER DAN DE BET365-PRIJS,
over de hele populatie wedstrijden - niet alleen over picks?

Opzet
-----
- Populatie: alle gespeelde wedstrijden met ECI-kansen en Bet365-odds uit
  betmobile_api_ready_mv. Standaard alleen vóór de freeze (2026-08-18);
  de wedstrijden daarna blijven bewust buiten beeld als latere toetsset.
- Per wedstrijd twee 'benen': THUIS en UIT (gelijkspel doet niet mee, het
  model zet daar nooit op in).
- Markt = laatste Bet365-snapshot vóór de aftrap (dezelfde closing-proxy als
  clv_report.py), Shin-gedevigd.
- Per kenmerk de vraag: weet dit cijfer iets wat de markt niet al weet?
      logit P(winst) = a + c * logit(p_markt) + b * kenmerk(gestandaardiseerd)
  b = 0: het kenmerk voegt niets toe. Cluster-robuuste SE (standaard per
  competitie), kritieke waarde t(G-1), plus een Bonferroni-grens voor het
  aantal geteste kenmerken.
- Stabiliteit: dezelfde toets in de eerste en de tweede helft van de periode.
  Een echt kenmerk wijst in beide helften dezelfde kant op.
- Lekcheck: de scraper overschrijft ratings en kansen in eci_data ook nog
  nádat een wedstrijd gespeeld is. Dan 'voorspelt' een kenmerk achteraf.
  Het script vergelijkt de db-waarden met de laatste snapshot van vóór de
  uitslag en markeert lekgevoelige kenmerken als dat nodig is.
- Als eci_history/ratings_history.jsonl bestaat, wordt het ratingverschil
  ook schoon berekend: laatste weekpunt strikt vóór de wedstrijddatum.

Verkennend en READ-ONLY. Verandert niets aan regels of database.

Gebruik
-------
    python research_populatie_kenmerken.py
    python research_populatie_kenmerken.py --cluster datum
    python research_populatie_kenmerken.py --odds view     # odds uit de view i.p.v. snapshots
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

# Werkt zowel vanuit de projectroot als vanuit eci_engine/.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE / "eci_engine", _HERE, _HERE.parent):
    if (_p / "config.py").exists() and (_p / "rules.py").exists():
        sys.path.insert(0, str(_p))
        break

from config import DB_DSN, OUTPUT_DIR
from db import get_upcoming_source_name
from prob_calibration import devig_shin

FREEZE_START = date(2026, 8, 18)
EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

# kolom, omschrijving, lekgevoelig voor ("ratings" / "kansen" / None)
KENMERKEN = [
    ("eci_kans",        "ECI-kans op winst (regel: min_prob)",                  "kansen"),
    ("verschil",        "ECI-kans min marktkans",                               "kansen"),
    ("value",           "odds x ECI-kans (regel: min_value)",                   "kansen"),
    ("rating_verschil", "eigen rating min tegenstander, uit de db",             "ratings"),
    ("rating_hist",     "idem, uit ratings_history (strikt vóór de wedstrijd)", None),
    ("drift",           "odds-beweging van eerste snapshot tot slot",           None),
    ("overround",       "marge van Bet365 (proxy voor marktdiepte)",            None),
]


def db_engine():
    """Read-only: dit script kan niets wijzigen in de database."""
    return create_engine(
        DB_DSN, connect_args={"options": "-c default_transaction_read_only=on"}
    )


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def parse_result(score) -> str | None:
    m = pd.Series(str(score)).str.extract(r"(\d+)\D+(\d+)").iloc[0]
    if m.isna().any():
        return None
    h, a = int(m.iloc[0]), int(m.iloc[1])
    return "H" if h > a else ("A" if a > h else "D")


def norm_name(s) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(s.lower().split())


# =====================================================================
# LADEN
# =====================================================================

POP_SQL = """
    WITH pop AS (
        SELECT v.match_id, v.date, v.competition, v.fixture_id, v.score,
               v.home_team, v.away_team,
               v.home_win_pct, v.draw_pct, v.away_win_pct,
               v.home_rating, v.away_rating,
               v.odds_home AS v_home, v.odds_draw AS v_draw, v.odds_away AS v_away,
               f.date_utc AS kickoff
        FROM public.{source} v
        JOIN public.fixtures f ON f.fixture_id = v.fixture_id
        WHERE v.score IS NOT NULL
          AND to_date(v.date, 'YYYY-MM-DD') >= :van
          AND to_date(v.date, 'YYYY-MM-DD') <  :tot
    ),
    snap AS (
        SELECT s.fixture_id, s.captured_at, s.odds_home, s.odds_draw, s.odds_away
        FROM public.odds_1x2_bet365_snapshots_mv s
        JOIN (SELECT DISTINCT fixture_id, kickoff FROM pop) p
          ON p.fixture_id = s.fixture_id
        WHERE s.captured_at <= p.kickoff
          AND s.odds_home > 1 AND s.odds_draw > 1 AND s.odds_away > 1
    ),
    slot AS (
        SELECT DISTINCT ON (fixture_id) fixture_id, captured_at AS slot_at,
               odds_home AS s_home, odds_draw AS s_draw, odds_away AS s_away
        FROM snap ORDER BY fixture_id, captured_at DESC
    ),
    eerste AS (
        SELECT DISTINCT ON (fixture_id) fixture_id, captured_at AS eerste_at,
               odds_home AS e_home, odds_draw AS e_draw, odds_away AS e_away
        FROM snap ORDER BY fixture_id, captured_at ASC
    )
    SELECT pop.*,
           slot.slot_at, slot.s_home, slot.s_draw, slot.s_away,
           eerste.eerste_at, eerste.e_home, eerste.e_draw, eerste.e_away
    FROM pop
    LEFT JOIN slot USING (fixture_id)
    LEFT JOIN eerste USING (fixture_id)
"""

LEK_SQL = """
    WITH pre AS (
        SELECT DISTINCT ON (match_id)
               match_id, home_rating, away_rating, home_win_pct, away_win_pct
        FROM public.eci_data_snapshots
        WHERE NULLIF(TRIM(eci_score), '') IS NULL
        ORDER BY match_id, snapshot_at DESC
    )
    SELECT v.score,
           v.home_rating  AS hr_db,  v.away_rating  AS ar_db,
           v.home_win_pct AS hp_db,  v.away_win_pct AS ap_db,
           pre.home_rating  AS hr_pre, pre.away_rating  AS ar_pre,
           pre.home_win_pct AS hp_pre, pre.away_win_pct AS ap_pre
    FROM public.{source} v
    JOIN pre ON pre.match_id = v.match_id
    WHERE v.score IS NOT NULL
"""


def load_population(source: str, van: date, tot: date) -> pd.DataFrame:
    with db_engine().connect() as conn:
        return pd.read_sql(text(POP_SQL.format(source=source)), conn,
                           params={"van": van, "tot": tot})


def load_lek(source: str) -> pd.DataFrame | None:
    try:
        with db_engine().connect() as conn:
            return pd.read_sql(text(LEK_SQL.format(source=source)), conn)
    except Exception as e:
        print(f"  lekcheck kon niet draaien: {str(e).splitlines()[0]}")
        return None


def load_rating_history():
    for p in (_HERE / "eci_history" / "ratings_history.jsonl",
              _HERE.parent / "eci_history" / "ratings_history.jsonl"):
        if p.exists():
            break
    else:
        return None
    df = pd.read_json(p, lines=True)
    df["date"] = pd.to_datetime(df["date"])
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_convert(None)
    df = df.sort_values(["eci_team_id", "date"])
    series = {str(t): (s["date"].values, s["points"].to_numpy(float))
              for t, s in df.groupby("eci_team_id")}
    names = {}
    for tid, name in df[["eci_team_id", "team_name"]].drop_duplicates().values:
        k = norm_name(name)
        if k:
            names.setdefault(k, str(tid))
    return series, names, p


def rating_before(series, tid, when) -> float:
    e = series.get(tid)
    if e is None:
        return np.nan
    dates, pts = e
    i = np.searchsorted(dates, np.datetime64(when), side="left")
    return float(pts[i - 1]) if i > 0 else np.nan


# =====================================================================
# VOORBEREIDEN
# =====================================================================

def prepare(df: pd.DataFrame, odds_bron: str, hist) -> tuple[pd.DataFrame, dict]:
    info = {"ruw": len(df)}
    df = df.drop_duplicates("fixture_id").copy()
    info["dubbel"] = info["ruw"] - len(df)

    num = ["home_win_pct", "draw_pct", "away_win_pct", "home_rating", "away_rating",
           "v_home", "v_draw", "v_away", "s_home", "s_draw", "s_away",
           "e_home", "e_draw", "e_away"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["home_win_pct", "draw_pct", "away_win_pct"]:
        if df[c].max() > 1.2:
            df[c] = df[c] / 100.0
    for c in ["kickoff", "slot_at", "eerste_at"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    df["result"] = df["score"].map(parse_result)
    df["competition"] = df["competition"].fillna("(onbekend)")

    info["met_slot"] = int(df["s_home"].notna().sum())
    mins = (df["kickoff"] - df["slot_at"]).dt.total_seconds() / 60
    info["slot_min_mediaan"] = float(mins.median()) if mins.notna().any() else np.nan

    src = {"slot": ("s_home", "s_draw", "s_away"), "view": ("v_home", "v_draw", "v_away")}[odds_bron]
    df["o_home"], df["o_draw"], df["o_away"] = df[src[0]], df[src[1]], df[src[2]]

    meer = df["eerste_at"] < df["slot_at"]
    df["drift_home"] = np.where(meer, df["s_home"] / df["e_home"] - 1, np.nan)
    df["drift_away"] = np.where(meer, df["s_away"] / df["e_away"] - 1, np.nan)

    # schone ratings uit de weekhistorie
    df["hist_home"] = np.nan
    df["hist_away"] = np.nan
    if hist is not None:
        series, names, _ = hist
        dates = pd.to_datetime(df["date"])
        hh, ha = [], []
        for h, a, d in zip(df["home_team"], df["away_team"], dates):
            hid, aid = names.get(norm_name(h)), names.get(norm_name(a))
            hh.append(rating_before(series, hid, d) if hid else np.nan)
            ha.append(rating_before(series, aid, d) if aid else np.nan)
        df["hist_home"], df["hist_away"] = hh, ha
    info["met_hist"] = int((df["hist_home"].notna() & df["hist_away"].notna()).sum())

    odds = df[["o_home", "o_draw", "o_away"]].to_numpy(float)
    ok = (np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
          & df["result"].notna().to_numpy()
          & df["home_win_pct"].notna().to_numpy() & df["away_win_pct"].notna().to_numpy())
    df = df[ok].copy()
    odds = odds[ok]
    mkt, _ = devig_shin(odds)
    df["mkt_home"], df["mkt_draw"], df["mkt_away"] = mkt[:, 0], mkt[:, 1], mkt[:, 2]
    df["overround"] = (1.0 / odds).sum(axis=1)
    info["bruikbaar"] = len(df)
    return df, info


def to_legs(df: pd.DataFrame) -> pd.DataFrame:
    stukken = []
    for kant, own, opp, code in (("THUIS", "home", "away", "H"), ("UIT", "away", "home", "A")):
        stukken.append(pd.DataFrame({
            "match_id": df["match_id"].values,
            "datum": pd.to_datetime(df["date"]).values,
            "competitie": df["competition"].values,
            "kant": kant,
            "odds": df[f"o_{own}"].values,
            "markt_kans": df[f"mkt_{own}"].values,
            "eci_kans": df[f"{own}_win_pct"].values,
            "rating_verschil": (df[f"{own}_rating"] - df[f"{opp}_rating"]).values,
            "rating_hist": (df[f"hist_{own}"] - df[f"hist_{opp}"]).values,
            "drift": df[f"drift_{own}"].values,
            "overround": df["overround"].values,
            "won": (df["result"] == code).astype(int).values,
        }))
    legs = pd.concat(stukken, ignore_index=True)
    legs["verschil"] = legs["eci_kans"] - legs["markt_kans"]
    legs["value"] = legs["odds"] * legs["eci_kans"]
    legs["datum_str"] = legs["datum"].dt.strftime("%Y-%m-%d")
    return legs


# =====================================================================
# STATISTIEK
# =====================================================================

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logit(y, X, clusters):
    """Logistische regressie (IRLS) met cluster-robuuste sandwich-SE."""
    b = np.zeros(X.shape[1])
    b[1] = 1.0                       # start: markt gekalibreerd
    for _ in range(100):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    bread = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    Sg = pd.DataFrame(X * (y - p)[:, None]).groupby(clusters).sum().to_numpy()
    G = Sg.shape[0]
    V = bread @ (Sg.T @ Sg) @ bread * (G / (G - 1)) if G > 1 else np.full_like(bread, np.nan)
    return b, np.sqrt(np.diag(V)), G, p


def toets(d: pd.DataFrame, col: str, cl: str) -> dict:
    d = d.dropna(subset=[col, "markt_kans"])
    if len(d) < 100 or d[col].std() == 0:
        return {"n": len(d), "t": np.nan, "pp_per_sd": np.nan, "G": 0}
    x = d[col].to_numpy(float)
    z = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones(len(d)), logit(d["markt_kans"]), z])
    b, se, G, p = fit_logit(d["won"].to_numpy(float), X, d[cl].to_numpy())
    return {"n": len(d), "G": G, "t": b[2] / se[2],
            "pp_per_sd": 100 * b[2] * np.mean(p * (1 - p))}


def cluster_mean(x: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(groups).sum().to_numpy()
    G = len(s)
    if G < 2:
        return m, np.nan
    return m, np.sqrt((s ** 2).sum()) / n * np.sqrt(G / (G - 1))


def fmt(v: float) -> str:
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def kwintielen(legs: pd.DataFrame, col: str, cl: str, q: int = 5) -> pd.DataFrame:
    d = legs.dropna(subset=[col]).copy()
    d["bucket"] = pd.qcut(d[col], q, duplicates="drop")
    rows = []
    for bkt, s in d.groupby("bucket", observed=True):
        m, se = cluster_mean((s["won"] - s["markt_kans"]).to_numpy(float), s[cl].to_numpy())
        winst = np.where(s["won"] == 1, s["odds"] - 1.0, -1.0)
        rows.append({
            "bereik": f"{fmt(bkt.left)} .. {fmt(bkt.right)}",
            "n": len(s),
            "gem_odds": round(s["odds"].mean(), 2),
            "markt%": round(100 * s["markt_kans"].mean(), 1),
            "echt%": round(100 * s["won"].mean(), 1),
            "verschil_pp": round(100 * m, 2),
            "se_pp": round(100 * se, 2),
            "roi%": round(100 * winst.mean(), 1),
        })
    return pd.DataFrame(rows)


# =====================================================================
# LEKCHECK
# =====================================================================

def samenvat_lek(diff: pd.Series, gewijzigd: pd.Series, res: pd.Series) -> dict:
    n = len(diff)
    if n < 30:
        return {"n": n, "oordeel": "te weinig data"}
    h, a = diff[res == "H"], diff[res == "A"]
    se = np.sqrt(h.var() / len(h) + a.var() / len(a)) if len(h) > 1 and len(a) > 1 else np.nan
    if se > 0:
        t = (h.mean() - a.mean()) / se
    elif h.mean() > a.mean():
        t = np.inf                   # volledig systematisch, zonder spreiding
    else:
        t = 0.0
    aandeel = float(gewijzigd.mean())
    if aandeel >= 0.05 and t > 3:
        oordeel = "LEK"
    elif aandeel >= 0.05 and t > 2:
        oordeel = "verdacht"
    else:
        oordeel = "schoon"
    return {"n": n, "aandeel": aandeel, "H": h.mean(), "D": diff[res == "D"].mean(),
            "A": a.mean(), "t": t, "oordeel": oordeel}


def beoordeel_lek(d: pd.DataFrame) -> dict:
    d = d.copy()
    for c in d.columns:
        if c != "score":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["hp_db", "ap_db", "hp_pre", "ap_pre"]:
        if d[c].max() > 1.2:
            d[c] = d[c] / 100.0
    d["res"] = d["score"].map(parse_result)
    d = d.dropna(subset=["res"])

    r = d.dropna(subset=["hr_db", "ar_db", "hr_pre", "ar_pre"])
    ratings = samenvat_lek(
        (r["hr_db"] - r["ar_db"]) - (r["hr_pre"] - r["ar_pre"]),
        ((r["hr_db"] - r["hr_pre"]).abs() >= 1) | ((r["ar_db"] - r["ar_pre"]).abs() >= 1),
        r["res"],
    )
    k = d.dropna(subset=["hp_db", "ap_db", "hp_pre", "ap_pre"])
    kansen = samenvat_lek(
        100 * (k["hp_db"] - k["hp_pre"]),
        ((k["hp_db"] - k["hp_pre"]).abs() > 0.005) | ((k["ap_db"] - k["ap_pre"]).abs() > 0.005),
        k["res"],
    )
    return {"ratings": ratings, "kansen": kansen}


def print_lek(lek: dict | None) -> dict:
    hr("LEKCHECK: veranderen ECI-waarden nog nadat de uitslag bekend is?")
    vlag = {"ratings": None, "kansen": None}
    if lek is None:
        print("  niet uitgevoerd; lekgevoelige kenmerken zijn ongecontroleerd")
        return vlag
    print("  vergelijkt de db-waarde met de laatste snapshot van vóór de uitslag")
    print("  (snapshots bestaan sinds eind augustus; de scraper werkt voor oudere")
    print("   wedstrijden op dezelfde manier, dus het oordeel geldt ook daarvoor)\n")
    for naam, eenheid in (("ratings", "ratingverschil thuis-uit, punten"),
                          ("kansen", "ECI-thuiskans, procentpunt")):
        s = lek[naam]
        if "aandeel" not in s:
            print(f"  {naam:8s}: {s['oordeel']} (n = {s['n']})")
            continue
        print(f"  {naam:8s}: {s['n']} wedstrijden, {100 * s['aandeel']:.0f}% gewijzigd na de uitslag")
        print(f"            verandering in {eenheid}: thuiswinst {s['H']:+.2f} | "
              f"gelijk {s['D']:+.2f} | uitwinst {s['A']:+.2f}   (t = {s['t']:.1f})"
              f"  -> {s['oordeel']}")
        vlag[naam] = s["oordeel"] in ("LEK", "verdacht")
    return vlag


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Welke cijfers verslaan de Bet365-prijs?")
    ap.add_argument("--van", default="2000-01-01")
    ap.add_argument("--tot", default=FREEZE_START.isoformat(),
                    help="alleen wedstrijden vóór deze datum (standaard: begin freeze)")
    ap.add_argument("--allow-freeze-data", action="store_true")
    ap.add_argument("--odds", choices=["slot", "view"], default="slot",
                    help="slot = laatste Bet365-snapshot vóór aftrap; view = odds uit de view")
    ap.add_argument("--cluster", choices=["competitie", "datum"], default="competitie")
    args = ap.parse_args()

    van = datetime.strptime(args.van, "%Y-%m-%d").date()
    tot = datetime.strptime(args.tot, "%Y-%m-%d").date()
    if tot > FREEZE_START and not args.allow_freeze_data:
        raise SystemExit(f"--tot {tot} valt na het begin van de freeze ({FREEZE_START}). "
                         "Geef --allow-freeze-data mee als dat echt de bedoeling is.")
    cl = "competitie" if args.cluster == "competitie" else "datum_str"

    source = get_upcoming_source_name()
    hr("POPULATIE")
    hist = load_rating_history()
    df, info = prepare(load_population(source, van, tot), args.odds, hist)
    legs = to_legs(df)

    print(f"  bron                 {source}, wedstrijden vóór {tot}")
    print(f"  gespeeld             {info['ruw']:,} (dubbele fixtures verwijderd: {info['dubbel']})")
    print(f"  met slotsnapshot     {info['met_slot']:,}  "
          f"(mediaan {info['slot_min_mediaan']:.0f} min vóór aftrap)")
    if hist is not None:
        print(f"  met ratinghistorie   {info['met_hist']:,}  ({hist[2].name})")
    else:
        print("  ratinghistorie       niet gevonden (eci_history/ratings_history.jsonl)")
    print(f"  bruikbaar            {info['bruikbaar']:,} wedstrijden = {len(legs):,} benen (thuis + uit)")
    print(f"  odds                 {args.odds}, cluster op {args.cluster}")
    if args.odds == "slot" and info["met_slot"] < 0.5 * max(info["ruw"], 1):
        print("  LET OP: minder dan de helft heeft een slotsnapshot. Overweeg --odds view.")

    vlag = print_lek(beoordeel_lek(d) if (d := load_lek(source)) is not None else None)

    # ---- markt ----------------------------------------------------------
    hr("MARKTCHECK: klopt de Bet365-slotkoers zelf?")
    X = np.column_stack([np.ones(len(legs)), logit(legs["markt_kans"])])
    b, se, G, _ = fit_logit(legs["won"].to_numpy(float), X, legs[cl].to_numpy())
    print(f"  helling c = {b[1]:.3f} (se {se[1]:.3f}), t tegen 1 = {(b[1] - 1) / se[1]:+.2f}")
    print("  c = 1: koers goed gekalibreerd. c > 1: favorieten winnen vaker dan de")
    print("  koers zegt. Het model hieronder corrigeert daar al voor.\n")
    print(kwintielen(legs, "markt_kans", cl).to_string(index=False))

    # ---- ranglijst ------------------------------------------------------
    actief = [(c, o, l) for c, o, l in KENMERKEN if legs[c].notna().sum() >= 100]
    grens = legs["datum"].median()
    rijen = []
    for col, oms, lekbron in actief:
        r = toets(legs, col, cl)
        r["kenmerk"] = col
        r["t_thuis"] = toets(legs[legs["kant"] == "THUIS"], col, cl)["t"]
        r["t_uit"] = toets(legs[legs["kant"] == "UIT"], col, cl)["t"]
        r["t_h1"] = toets(legs[legs["datum"] <= grens], col, cl)["t"]
        r["t_h2"] = toets(legs[legs["datum"] > grens], col, cl)["t"]
        r["lek"] = bool(lekbron and vlag.get(lekbron))
        rijen.append(r)
    rang = pd.DataFrame(rijen)
    G = int(rang["G"].max())
    k = len(rang)
    t_crit = stats.t.ppf(0.975, G - 1)
    t_bonf = stats.t.ppf(1 - 0.025 / k, G - 1)

    def oordeel(r):
        if r["lek"]:
            return "LEK?"
        zelfde = np.sign(r["t_h1"]) == np.sign(r["t"]) == np.sign(r["t_h2"])
        if abs(r["t"]) > t_bonf and zelfde:
            return "SIGNAAL"
        if abs(r["t"]) > t_crit:
            return "zwak"
        return "-"

    rang["oordeel"] = rang.apply(oordeel, axis=1)
    rang = rang.reindex(rang["t"].abs().sort_values(ascending=False).index)

    hr("RANGLIJST: welk cijfer voorspelt de uitslag beter dan de Bet365-prijs?")
    toon = rang[["kenmerk", "n", "pp_per_sd", "t", "t_thuis", "t_uit", "t_h1", "t_h2", "oordeel"]]
    print(toon.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(f"\n  pp_per_sd : extra winkans in procentpunt per standaardafwijking van het")
    print(f"              kenmerk, bovenop de (gekalibreerde) marktkans")
    print(f"  SIGNAAL   : |t| > {t_bonf:.2f} (gecorrigeerd voor {k} kenmerken) én in beide")
    print(f"              helften dezelfde kant op. zwak: alleen |t| > {t_crit:.2f}.")
    print(f"  LEK?      : kenmerk hangt af van db-waarden die na de uitslag veranderen")
    print(f"  clusters  : {G} ({args.cluster}); helften geknipt op {grens:%Y-%m-%d}")

    # ---- kwintielen -----------------------------------------------------
    hr("KWINTIELEN per kenmerk (verschil_pp = echt% min markt%)")
    for col, oms, _ in actief:
        print(f"\n--- {col}: {oms}")
        print(kwintielen(legs, col, cl).to_string(index=False))

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    p1 = EXPORT_DIR / f"populatie_ranglijst_{stamp}.csv"
    p2 = EXPORT_DIR / f"populatie_benen_{stamp}.csv"
    rang.to_csv(p1, index=False)
    legs.drop(columns=["datum_str"]).to_csv(p2, index=False)
    print(f"\n[export] {p1}\n[export] {p2}")
    print("\nKlaar. Read-only: er is niets gewijzigd.")


if __name__ == "__main__":
    main()