"""
toets_prijsbeweging.py
======================

VOORAF VASTGELEGDE BEVESTIGINGSTOETS - opzet en beslisregel staan in
PREREG_PRIJSBEWEGING.md (vastgelegd 23-10-2026). Dit bestand niet meer
aanpassen. Elke run print de SHA-256 van dit bestand (regeleinden
genormaliseerd), zodat te controleren is dat precies de vastgelegde versie
heeft gedraaid.

VRAAG
  Voorspelt de ECI-selectie een gunstiger prijsbeweging na instap dan
  vergelijkbare kandidaten zonder ECI-selectie?

WANNEER
  Pas als de meetperiode het eindpunt uit MEETAFSPRAAK.md heeft bereikt:
  150 afgerekende MAIN-picks met wedstrijddatum vanaf 18-08-2026. Tot dan
  weigert het script en toont het alleen hoeveel MAIN-picks er zijn
  afgerekend. Er is geen optie om dat te omzeilen.

WAT HET DOET
  - kandidaten: elke (wedstrijd, kant) bij de eerste modelrun waarop het
    marktdeel van de bevroren regels geldt (odds 1,50-2,50, >= 7 snapshots,
    drift <= +3%); wedstrijddatum 18-08-2026 t/m de datum van de 150e
    afgerekende MAIN-pick
  - ECI-selectie: de bevroren MAIN-regel kiest die kant OP DATZELFDE
    instapmoment (geen vooruitkijken)
  - uitkomst: beweging = (p_close - p_instap) x odds, Shin-kansen, slotkoers
    = laatste Bet365-snapshot op of voor de aftrap
  - toets: OLS van beweging op ECI-selectie plus oddsband-dummies,
    cluster-robuust per competitie, t(G-1), tweezijdig 95%
  - relevant: verschil van minstens +1,0 procentpunt

Geen ROI en geen uitslagen: de prestaties van de selectie zelf worden
beoordeeld via MEETAFSPRAAK.md en apart gerapporteerd.

READ-ONLY. Wijzigt niets aan config, regels of database.

Gebruik (vanuit de projectmap):
    python toets_prijsbeweging.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if (_p / "betmobile_settings.py").exists():
        sys.path.insert(0, str(_p))
        break

# ---------------------------------------------------------------------
# VASTGELEGD OP 23-10-2026 - NIET WIJZIGEN
# ---------------------------------------------------------------------
FREEZE_START = date(2026, 8, 18)
N_EINDPUNT = 150                     # afgerekende MAIN-picks, zie MEETAFSPRAAK.md
REGELS = {"min_prob": 0.58, "min_value": 1.02, "min_rating_gap": 500,
          "min_odds": 1.50, "max_odds": 2.50, "min_snapshots": 7, "min_drift_abs": 0.0}
MIN_STRENGTH = 1.8
DRIFT_OPPOSE = 0.03
ODDSBANDEN = [1.50, 1.65, 1.80, 2.00, 2.5001]
BAND_LABELS = ["1.50-1.65", "1.65-1.80", "1.80-2.00", "2.00-2.50"]
RELEVANT = 0.010                     # +1,0 procentpunt beweging
ALPHA = 0.05                         # tweezijdig
EXACT_DREMPEL = 0.995
LINK_KANDIDATEN = ["eci_fixture_link_mv", "eci_fixture_links_v", "eci_fixture_link_v"]
NODIG = [
    "run_id", "match_id", "date", "competition",
    "odds_home", "odds_draw", "odds_away", "prob_home", "prob_away",
    "bet_home", "bet_away", "rating_gap", "rating_home_edge", "n_snapshots",
    "home_drift_pct", "away_drift_pct", "home_rule", "away_rule",
    "rule_strength_adj_home", "rule_strength_adj_away",
    "rule_strength_calibrated_home", "rule_strength_calibrated_away",
    "is_pick", "pick_type", "selection",
]


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def eigen_hash() -> str:
    data = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def engine():
    from betmobile_settings import DB_DSN
    return create_engine(DB_DSN, connect_args={"options": "-c default_transaction_read_only=on"})


def lees(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


# ---------------------------------------------------------------------
# Shin (kopie van prob_calibration.devig_shin, bevroren op 23-10-2026)
# ---------------------------------------------------------------------

def _shin_z(imp: np.ndarray, max_iter: int = 60, tol: float = 1e-10) -> np.ndarray:
    total = imp.sum(axis=1)
    z = np.zeros(len(imp))
    k = imp.shape[1]
    for _ in range(max_iter):
        disc = z[:, None] ** 2 + 4.0 * (1.0 - z)[:, None] * (imp ** 2) / total[:, None]
        p = (np.sqrt(np.clip(disc, 0, None)) - z[:, None]) / (2.0 * np.clip(1.0 - z, 1e-9, None)[:, None])
        z_new = np.clip(z + (p.sum(axis=1) - 1.0) / max(k - 1.0, 1e-9), 0.0, 0.9)
        if np.max(np.abs(z_new - z)) < tol:
            z = z_new
            break
        z = z_new
    return z


def shin(odds: np.ndarray) -> np.ndarray:
    """Shin-kansen per rij; rijen met een ongeldige odd (<= 1,01) krijgen NaN."""
    odds = np.asarray(odds, dtype=float)
    ok = np.isfinite(odds).all(axis=1) & (odds > 1.01).all(axis=1)
    out = np.full(odds.shape, np.nan)
    if ok.any():
        imp = 1.0 / odds[ok]
        total = imp.sum(axis=1)
        z = _shin_z(imp)
        disc = z[:, None] ** 2 + 4.0 * (1.0 - z)[:, None] * (imp ** 2) / total[:, None]
        p = (np.sqrt(np.clip(disc, 0, None)) - z[:, None]) / (2.0 * np.clip(1.0 - z, 1e-9, None)[:, None])
        out[ok] = p / p.sum(axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------
# Eindpunt van de meetperiode
# ---------------------------------------------------------------------

def eindpunt() -> tuple[int, date | None]:
    """Aantal afgerekende MAIN-picks vanaf 18-08 en de datum van de 150e.

    Leest alleen datum en aantal; geen uitslagen of rendement.
    """
    d = lees("""
        SELECT match_id, selection,
               COALESCE(date_ts, NULLIF(date::text, '')::date::timestamp)::date AS wd
        FROM public.picks_evaluated_unique_v
        WHERE pick_type = 'MAIN'
          AND selection IS NOT NULL
          AND outcome IN ('WIN', 'LOSS')
          AND COALESCE(date_ts, NULLIF(date::text, '')::date::timestamp) >= :start
        ORDER BY wd, match_id, selection
    """, {"start": FREEZE_START})
    if len(d) < N_EINDPUNT:
        return len(d), None
    return len(d), pd.to_datetime(d["wd"].iloc[N_EINDPUNT - 1]).date()


# ---------------------------------------------------------------------
# Data en regels
# ---------------------------------------------------------------------

def laad_runs(tot: date) -> pd.DataFrame:
    cols = lees("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'model_match_snapshots'
    """)["column_name"].tolist()
    ontbreekt = [c for c in NODIG if c not in cols]
    if ontbreekt:
        raise SystemExit(f"model_match_snapshots mist kolommen {ontbreekt}; toets niet uitvoerbaar.")
    df = lees(f"""
        SELECT {", ".join(NODIG)} FROM public.model_match_snapshots
        WHERE LEFT(date::text, 10)::date BETWEEN :van AND :tot
    """, {"van": FREEZE_START, "tot": tot})
    for c in NODIG:
        if c not in ("match_id", "date", "competition", "home_rule", "away_rule",
                     "is_pick", "pick_type", "selection"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("prob_home", "prob_away"):
        if df[c].max() > 1.2:
            df[c] = df[c] / 100.0
    for c in ("home_rule", "away_rule", "is_pick"):
        df[c] = df[c].map(lambda v: bool(v) if pd.notna(v) else False)
    df["competition"] = df["competition"].fillna("(onbekend)")
    df["pick_type"] = df["pick_type"].fillna("")
    df["selection"] = df["selection"].fillna("")
    return df.drop_duplicates(["run_id", "match_id"]).reset_index(drop=True)


def marktdeel(df: pd.DataFrame, side: str) -> pd.Series:
    drift = df[f"{side}_drift_pct"]
    return (df[f"odds_{side}"].between(REGELS["min_odds"], REGELS["max_odds"])
            & (df["n_snapshots"].fillna(0) >= REGELS["min_snapshots"])
            & (drift.abs().fillna(0) >= REGELS["min_drift_abs"])
            & (drift.fillna(0) <= DRIFT_OPPOSE))


def herbereken_vlag(df: pd.DataFrame, side: str) -> pd.Series:
    ok = (marktdeel(df, side)
          & (df[f"prob_{side}"] >= REGELS["min_prob"])
          & (df[f"bet_{side}"] >= REGELS["min_value"])
          & (df["rating_gap"] >= REGELS["min_rating_gap"]))
    if side == "home":
        ok = ok & (df["rating_home_edge"] >= 0)
    return ok


def kies_main(df: pd.DataFrame, sterkte: str) -> np.ndarray:
    h_ok = df["home_rule"].to_numpy(bool)
    a_ok = df["away_rule"].to_numpy(bool)
    sh = df[f"rule_strength_{sterkte}_home"].fillna(0).to_numpy(float)
    sa = df[f"rule_strength_{sterkte}_away"].fillna(0).to_numpy(float)
    sel = np.full(len(df), "", dtype=object)
    sel[a_ok & ~h_ok] = "AWAY"
    sel[h_ok & ~a_ok] = "HOME"
    beide = h_ok & a_ok
    sel[beide] = np.where(sa[beide] >= sh[beide], "AWAY", "HOME")
    sterk = np.where(sel == "HOME", sh, np.where(sel == "AWAY", sa, np.nan))
    sel[~((sel != "") & (sterk >= MIN_STRENGTH))] = ""
    return sel


def controle_reconstructie(df: pd.DataFrame) -> bool:
    hr("STAP 1 - RECONSTRUCTIE VAN DE SELECTIE")
    akkoord = []
    for side in ("home", "away"):
        gelijk = float((herbereken_vlag(df, side) == df[f"{side}_rule"]).mean())
        akkoord.append(gelijk)
        print(f"  regelvlag {side:5s}: {100 * gelijk:7.3f}% gelijk aan opgeslagen")
    na = kies_main(df, "calibrated")         # wat de engine per run gebruikte
    echt = np.where(df["is_pick"].to_numpy(bool) & (df["pick_type"].to_numpy() == "MAIN"),
                    df["selection"].to_numpy(), "")
    tp = int(((na != "") & (na == echt)).sum())
    rest = int(((na != "") ^ (echt != "")).sum() + ((na != "") & (echt != "") & (na != echt)).sum())
    pick_ok = tp / (tp + rest) if (tp + rest) else 1.0
    print(f"  MAIN-picks nagebootst: {100 * pick_ok:7.3f}% ({tp} gelijk, {rest} afwijkend)")
    geldig = min(akkoord) >= EXACT_DREMPEL and pick_ok >= EXACT_DREMPEL
    print(f"  -> {'reconstructie geldig' if geldig else 'RECONSTRUCTIE ONGELDIG: toets niet uitgevoerd'}")
    return geldig


# ---------------------------------------------------------------------
# Koppelingen, slotkoers en uitslag
# ---------------------------------------------------------------------

def laad_link() -> pd.DataFrame:
    for naam in LINK_KANDIDATEN:
        cols = lees("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = :t
        """, {"t": naam})["column_name"].tolist()
        eci = next((c for c in ("match_id", "eci_match_id") if c in cols), None)
        if eci and "fixture_id" in cols:
            d = lees(f"SELECT {eci} AS match_id, fixture_id FROM public.{naam}")
            return d.dropna().drop_duplicates("match_id")
    raise SystemExit("Geen linkview met match_id en fixture_id gevonden.")


def laad_slot_en_uitslag(ids: list[int]) -> pd.DataFrame:
    lijst = ",".join(str(int(i)) for i in ids) or "NULL"
    snaps = lees(f"""
        SELECT s.fixture_id, s.captured_at, s.odds_home, s.odds_draw, s.odds_away, f.date_utc AS kickoff_at
        FROM public.odds_1x2_bet365_snapshots_mv s
        JOIN public.fixtures f ON f.fixture_id = s.fixture_id
        WHERE s.fixture_id IN ({lijst})
    """)
    snaps["captured_at"] = pd.to_datetime(snaps["captured_at"], utc=True, errors="coerce")
    snaps["kickoff_at"] = pd.to_datetime(snaps["kickoff_at"], utc=True, errors="coerce")
    snaps = snaps[snaps["captured_at"] <= snaps["kickoff_at"]]
    slot = snaps.sort_values("captured_at").groupby("fixture_id", as_index=False).last()
    p = shin(slot[["odds_home", "odds_draw", "odds_away"]].to_numpy(float))
    slot["pc_home"], slot["pc_away"] = p[:, 0], p[:, 2]
    uitslag = lees(f"""
        SELECT fixture_id, ft_home_goals, ft_away_goals, status_short
        FROM public.fixtures WHERE fixture_id IN ({lijst})
    """)
    return slot[["fixture_id", "kickoff_at", "pc_home", "pc_away"]].merge(uitslag, on="fixture_id", how="outer")


# ---------------------------------------------------------------------
# Statistiek
# ---------------------------------------------------------------------

def ols_cluster(y: np.ndarray, X: np.ndarray, groepen: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    codes = pd.factorize(pd.Series(groepen))[0]
    G = int(codes.max()) + 1
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in range(G):
        m = codes == g
        u = X[m].T @ r[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    return b, np.sqrt(np.diag(V)), G


def cluster_mean(x: np.ndarray, groepen: np.ndarray) -> tuple[float, float, int]:
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    m = float(x.mean())
    s = pd.Series(x - m).groupby(groepen).sum().to_numpy()
    G = len(s)
    if G < 2:
        return m, np.nan, G
    return m, float(np.sqrt((s ** 2).sum()) / n * np.sqrt(G / (G - 1))), G


def fmt_int(m: float, se: float, G: int) -> str:
    if not np.isfinite(se) or G < 2:
        return f"{100 * m:+6.2f} (n.v.t.)"
    c = stats.t.ppf(1 - ALPHA / 2, G - 1)
    return f"{100 * m:+6.2f} [{100 * (m - c * se):+6.2f}; {100 * (m + c * se):+6.2f}]"


# ---------------------------------------------------------------------

def main() -> None:
    hr("BEVESTIGINGSTOETS PRIJSBEWEGING (PREREG_PRIJSBEWEGING.md)")
    print(f"  SHA-256 van dit script: {eigen_hash()}")
    print("  Vergelijk met de hash in PREREG_PRIJSBEWEGING.md; wijkt die af, dan is dit")
    print("  niet de vastgelegde versie en telt de uitkomst niet als bevestiging.")

    n, tot = eindpunt()
    print(f"\n  afgerekende MAIN-picks vanaf {FREEZE_START}: {n} van {N_EINDPUNT}")
    if tot is None:
        print("  Het eindpunt is nog niet bereikt. De toets draait pas bij "
              f"{N_EINDPUNT} afgerekende MAIN-picks.")
        return
    print(f"  eindpunt bereikt: wedstrijddatum van de {N_EINDPUNT}e pick is {tot}")
    print(f"  toetsperiode: wedstrijddatum {FREEZE_START} t/m {tot}")

    df = laad_runs(tot)
    print(f"  {len(df):,} rijen (run x wedstrijd), {df['match_id'].nunique():,} wedstrijden")
    if not controle_reconstructie(df):
        return

    bevroren = kies_main(df, "adj")          # bevroren config: calibratie uit
    stukken = []
    for side, code in (("home", "HOME"), ("away", "AWAY")):
        stukken.append(pd.DataFrame({
            "run_id": df["run_id"].values, "match_id": df["match_id"].values,
            "competition": df["competition"].values, "kant": code,
            "odds": df[f"odds_{side}"].values,
            "o_h": df["odds_home"].values, "o_d": df["odds_draw"].values, "o_a": df["odds_away"].values,
            "B": marktdeel(df, side).values, "ECI": (bevroren == code),
        }))
    lang = pd.concat(stukken, ignore_index=True)
    kand = (lang[lang["B"]].sort_values("run_id")
            .drop_duplicates(["match_id", "kant"], keep="first").copy())

    link = laad_link()
    kand["fixture_id"] = kand["match_id"].map(link.set_index("match_id")["fixture_id"])
    ids = sorted({int(i) for i in kand["fixture_id"].dropna()})
    kand = kand.merge(laad_slot_en_uitslag(ids), on="fixture_id", how="left")

    p_in = shin(kand[["o_h", "o_d", "o_a"]].to_numpy(float))
    kand["p_instap"] = np.where(kand["kant"] == "HOME", p_in[:, 0], p_in[:, 2])
    kand["p_close"] = np.where(kand["kant"] == "HOME", kand["pc_home"], kand["pc_away"])
    kand["beweging"] = (kand["p_close"] - kand["p_instap"]) * kand["odds"]
    kand["marge_instap"] = kand["p_instap"] * kand["odds"] - 1.0
    kand["clv"] = kand["p_close"] * kand["odds"] - 1.0
    gespeeld = kand["status_short"].isin(["FT", "AET", "PEN"]) & kand["ft_home_goals"].notna()
    ok = kand[gespeeld & kand["p_close"].notna() & kand["p_instap"].notna() & kand["odds"].notna()].copy()
    ok["band"] = pd.cut(ok["odds"], ODDSBANDEN, labels=BAND_LABELS, right=False)
    ok = ok[ok["band"].notna()]

    hr("STAP 2 - PRIMAIRE TOETS")
    eci, rest = ok[ok["ECI"]], ok[~ok["ECI"]]
    print(f"  kandidaten: {len(ok)} (ECI-selectie op instap {len(eci)}, overige {len(rest)}); "
          f"uitgevallen zonder slotkoers/uitslag/instapodds: {len(kand) - len(ok)}")
    if len(eci) < 2 or len(rest) < 2:
        print("  Te weinig kandidaten in een van beide groepen; toets niet uitvoerbaar.")
        return
    dummies = pd.get_dummies(ok["band"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(ok)), ok["ECI"].to_numpy(float), dummies.to_numpy(float)])
    b, se, G = ols_cluster(ok["beweging"].to_numpy(float), X, ok["competition"].to_numpy())
    crit = stats.t.ppf(1 - ALPHA / 2, G - 1)
    lo, hi = b[1] - crit * se[1], b[1] + crit * se[1]
    print("  verschil in beweging (ECI - overige), gecorrigeerd voor oddsband:")
    print(f"    {100 * b[1]:+.2f} procentpunt, 95%-interval [{100 * lo:+.2f}; {100 * hi:+.2f}], "
          f"t = {b[1] / se[1]:.2f}, G = {G}, kritiek t({G - 1}) = {crit:.3f}")

    if lo > 0 and b[1] >= RELEVANT:
        oordeel = "BEVESTIGD: positief en minstens zo groot als de relevantiegrens"
    elif lo > 0:
        oordeel = "AANGETOOND, MAAR KLEINER DAN DE RELEVANTIEGRENS"
    elif hi < RELEVANT:
        oordeel = "GEEN RELEVANT EFFECT: het interval sluit +1,0 procentpunt uit"
    else:
        oordeel = "ONBESLIST: het interval bevat zowel nul als de relevantiegrens"
    print(f"\n  >>> OORDEEL: {oordeel}")

    hr("STAP 3 - BESCHRIJVEND (TELT NIET MEE IN HET OORDEEL)")
    print(f"  {'groep':22s} {'n':>5s}  {'beweging %':>26s}  {'marge bij instap %':>26s}  {'CLV %':>26s}")
    for naam, g in (("ECI-selectie op instap", eci), ("overige kandidaten", rest), ("alle kandidaten", ok)):
        cl = g["competition"].to_numpy()
        cols = [fmt_int(*cluster_mean(g[k].to_numpy(float), cl)) for k in ("beweging", "marge_instap", "clv")]
        print(f"  {naam:22s} {len(g):5d}  {cols[0]:>26s}  {cols[1]:>26s}  {cols[2]:>26s}")
    print("\n  beweging per oddsband (gemiddelde in %):")
    for band, s in ok.groupby("band", observed=False):
        a, z = s[s["ECI"]], s[~s["ECI"]]
        ma = f"{100 * a['beweging'].mean():+.2f}" if len(a) else "  n.v.t."
        mz = f"{100 * z['beweging'].mean():+.2f}" if len(z) else "  n.v.t."
        print(f"    {band}: ECI n={len(a):3d} {ma:>7s}   overige n={len(z):4d} {mz:>7s}")

    print("\nKlaar. Read-only: er is niets gewijzigd. Rapporteer dit apart van het oordeel")
    print("over de 150 MAIN-picks (MEETAFSPRAAK.md).")


if __name__ == "__main__":
    main()
