"""
research_selectie_vergelijking.py
=================================

Helpt ECI om uit de wedstrijden die de markt aantrekkelijk maakt een
betere selectie te maken?

TWEE SELECTIEMETHODEN, OP DEZELFDE MODELRUNS
  A  de bevroren regels (MAIN-picks): het marktdeel hieronder PLUS de
     ECI-criteria (kans, value, rating gap, thuis-edge, strength).
  B  alleen het marktdeel van dezelfde regels: odds binnen de band,
     minimaal aantal snapshots, drift niet tegen. Geen ECI-criteria.

Elke (wedstrijd, kant) telt op het EERSTE moment dat zijn methode afgaat,
tegen de odds die het model op dat moment zag.

GROEPEN
  A           alle A-picks, op hun eigen moment (zoals je ze echt speelt)
  Beide       B-kandidaten die A ook kiest, op het B-moment
  Alleen B    B-kandidaten die A niet kiest, op het B-moment
  B geheel    alle B-kandidaten, op het B-moment

MATEN PER GROEP
  CLV        p_close x odds_genomen - 1, p_close = Shin-gedevigde kans uit
             de laatste Bet365-snapshot op of voor de aftrap (clv_report.py)
  uitslag    gewonnen - p_close: wint de groep vaker dan de slotkoers zegt?
  ROI        rendement tegen de genomen odds
  Per bet geldt: ROI = CLV + (uitslag - slotkans) x odds.
  Intervallen: 95%, cluster-robuust per competitie, t(G-1).

VERSCHILLEN
  Beide - Alleen B   zelfde moment: wat sluiten de ECI-filters uit?
  A - Alleen B       de praktijk: ECI-picks op hun eigen moment tegen de
                     kandidaten die ECI laat liggen
  A - Beide          zelfde kandidaten, ander moment: wat kost of
                     oplevert het wachten op het ECI-signaal (alleen CLV)

DEEL 3 (toegevoegd 23-10-2026, na de eerste run, met vooraf vastgelegde leesregel)
  CLV = marge bij instap + beweging tot de slotkoers
    marge bij instap = p_instap x odds - 1   (Shin, prijs op het instapmoment)
    beweging         = (p_close - p_instap) x odds
  De marge hangt af van prijsniveau en competitie (Shin legt meer marge op
  hogere odds); de beweging is wat de markt na instap deed. Het verschil
  Beide - Alleen B wordt voor beide delen getoond, en voor de beweging ook
  gecorrigeerd voor oddsband.

CONTROLE DEEL 3 (toegevoegd 23-10-2026): rekent per kandidaat na of
  CLV = marge + beweging en beweging = (p_close - p_instap) x odds kloppen,
  toont de groepsgemiddelden onafgerond, en telt hoeveel Beide-kandidaten A al
  op het B-instapmoment koos. Verandert geen uitkomst van deel 1-3.

PERIODE: VAST, 1 juli t/m 17 augustus 2026
  - Drempels gelden sinds 25 juni 2026 (git c55f37a), sindsdien ongewijzigd.
    De config is ontwikkeld op Unibet/Oddspedia t/m mei 2026 en Bet365-picks
    april-juni 2026.
  - De meetperiode (vanaf 18 augustus) blijft volledig buiten deze analyse:
    het script laadt geen wedstrijd vanaf die datum en heeft geen optie om
    dat te veranderen. Zie MEETAFSPRAAK.md.

EERST: KAN DE SELECTIE EXACT WORDEN GERECONSTRUEERD?
In deze periode stond de strength-calibratie nog aan; de bevroren config
heeft die uit. De regels worden daarom opnieuw toegepast op de per run
opgeslagen waarden (model_match_snapshots). Eerst wordt gecontroleerd:
  1  dekking van de benodigde kolommen
  2  regelvlaggen opnieuw berekend uit de opgeslagen invoer == opgeslagen vlaggen
  3  de destijds gemaakte MAIN-picks exact na te bootsen met de destijds
     gebruikte (gecalibreerde) strength
  4  hoeveel picks verandert het uitzetten van de calibratie
Zijn 2 of 3 niet (vrijwel) exact, dan stopt het script, tenzij --forceer.

READ-ONLY. Wijzigt niets aan config, regels of database.

Gebruik (vanuit de projectmap of vanuit eci_engine/):
    python research_selectie_vergelijking.py
    python research_selectie_vergelijking.py --export-csv
"""

from __future__ import annotations

import argparse
import sys
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

import config as C  # noqa: E402
from prob_calibration import compute_market_probs  # noqa: E402
from clv_report import load_link, load_kickoffs, load_snapshots, build_closing  # noqa: E402

VAN = date(2026, 7, 1)
TOT = date(2026, 8, 17)              # inclusief
FREEZE_START = date(2026, 8, 18)
assert TOT < FREEZE_START, "de analyseperiode mag de meetperiode niet raken"

# De bevroren waarden, ter controle op config.py
BEVROREN = {"min_prob": 0.58, "min_value": 1.02, "min_rating_gap": 500,
            "min_odds": 1.50, "max_odds": 2.50, "min_snapshots": 7, "min_drift_abs": 0}
BEVROREN_MIN_STRENGTH = 1.8
BEVROREN_DRIFT_OPPOSE = 0.03

EXACT_DREMPEL = 0.995                # minimale overeenstemming voor "vrijwel exact"

NODIG = [
    "run_id", "match_id", "date", "competition",
    "odds_home", "odds_draw", "odds_away", "prob_home", "prob_away",
    "bet_home", "bet_away", "rating_gap", "rating_home_edge", "n_snapshots",
    "hours_to_kickoff", "home_drift_pct", "away_drift_pct",
    "home_rule", "away_rule",
    "rule_strength_adj_home", "rule_strength_adj_away",
    "rule_strength_calibrated_home", "rule_strength_calibrated_away",
    "is_pick", "pick_type", "selection",
]

EXPORT_DIR = C.OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def hr(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def db_engine():
    """Read-only: dit script kan niets wijzigen in de database."""
    return create_engine(C.DB_DSN, connect_args={"options": "-c default_transaction_read_only=on"})


# =====================================================================
# LADEN
# =====================================================================

def load_snapshot_rows() -> tuple[pd.DataFrame, list[str]]:
    with db_engine().connect() as conn:
        cols = pd.read_sql(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'model_match_snapshots'
        """), conn)["column_name"].tolist()
        aanwezig = [c for c in NODIG if c in cols]
        ontbreekt = [c for c in NODIG if c not in cols]
        q = f"""
            SELECT {", ".join(aanwezig)}
            FROM public.model_match_snapshots
            WHERE LEFT(date::text, 10)::date BETWEEN :van AND :tot
        """
        df = pd.read_sql(text(q), conn, params={"van": VAN, "tot": TOT})
    return df, ontbreekt


def load_results(fixture_ids: list[int]) -> pd.DataFrame:
    ids = ",".join(str(int(i)) for i in fixture_ids) or "NULL"
    with db_engine().connect() as conn:
        return pd.read_sql(text(f"""
            SELECT fixture_id, ft_home_goals, ft_away_goals, status_short
            FROM public.fixtures WHERE fixture_id IN ({ids})
        """), conn)


# =====================================================================
# CONTROLE CONFIG
# =====================================================================

def controleer_config() -> None:
    afwijking = {k: (C.ECI_RULE_PARAMS.get(k), v) for k, v in BEVROREN.items()
                 if C.ECI_RULE_PARAMS.get(k) != v}
    if C.MIN_STRENGTH != BEVROREN_MIN_STRENGTH:
        afwijking["MIN_STRENGTH"] = (C.MIN_STRENGTH, BEVROREN_MIN_STRENGTH)
    if C.DRIFT_OPPOSE_THRESHOLD != BEVROREN_DRIFT_OPPOSE:
        afwijking["DRIFT_OPPOSE_THRESHOLD"] = (C.DRIFT_OPPOSE_THRESHOLD, BEVROREN_DRIFT_OPPOSE)
    if getattr(C, "USE_STRENGTH_CALIBRATION", False) or getattr(C, "USE_PROB_CALIBRATION", False):
        afwijking["calibratie"] = ("aan", "uit")
    if afwijking:
        raise SystemExit(f"config.py wijkt af van de bevroren configuratie: {afwijking}")
    p = C.ECI_RULE_PARAMS
    print(f"  bevroren config bevestigd: kans >= {p['min_prob']}, value >= {p['min_value']}, "
          f"gap >= {p['min_rating_gap']}, odds {p['min_odds']}-{p['max_odds']}, "
          f"snapshots >= {p['min_snapshots']}, drift <= +{C.DRIFT_OPPOSE_THRESHOLD}, "
          f"strength >= {C.MIN_STRENGTH}, calibratie uit")


# =====================================================================
# VOORBEREIDEN EN REGELS
# =====================================================================

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = ["odds_home", "odds_draw", "odds_away", "prob_home", "prob_away", "bet_home",
           "bet_away", "rating_gap", "rating_home_edge", "n_snapshots", "hours_to_kickoff",
           "home_drift_pct", "away_drift_pct", "rule_strength_adj_home", "rule_strength_adj_away",
           "rule_strength_calibrated_home", "rule_strength_calibrated_away", "run_id"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["prob_home", "prob_away"]:
        if c in df.columns and df[c].max() > 1.2:
            df[c] = df[c] / 100.0
    for c in ["home_rule", "away_rule", "is_pick"]:
        if c in df.columns:
            df[c] = df[c].map(lambda v: bool(v) if pd.notna(v) else False)
    df["competition"] = df["competition"].fillna("(onbekend)")
    df["pick_type"] = df.get("pick_type", pd.Series(index=df.index, dtype=object)).fillna("")
    df["selection"] = df.get("selection", pd.Series(index=df.index, dtype=object)).fillna("")
    return df


def marktdeel(df: pd.DataFrame, side: str) -> pd.Series:
    """Het marktdeel van de regel (zelfde condities als rules.py)."""
    p = C.ECI_RULE_PARAMS
    drift = df[f"{side}_drift_pct"]
    return (df[f"odds_{side}"].between(p["min_odds"], p["max_odds"])
            & (df["n_snapshots"].fillna(0) >= p["min_snapshots"])
            & (drift.abs().fillna(0) >= p["min_drift_abs"])
            & (drift.fillna(0) <= C.DRIFT_OPPOSE_THRESHOLD))


def herbereken_vlag(df: pd.DataFrame, side: str) -> pd.Series:
    """Volledige regel per kant, opnieuw berekend uit de opgeslagen invoer."""
    p = C.ECI_RULE_PARAMS
    ok = (marktdeel(df, side)
          & (df[f"prob_{side}"] >= p["min_prob"])
          & (df[f"bet_{side}"] >= p["min_value"])
          & (df["rating_gap"] >= p["min_rating_gap"]))
    if side == "home":
        ok = ok & (df["rating_home_edge"] >= 0)
    return ok


def kies_main(df: pd.DataFrame, sterkte: str) -> np.ndarray:
    """MAIN-selectie zoals picks.py: kant kiezen, daarna strength-filter.

    sterkte = "calibrated" bootst de destijds gemaakte picks na,
              "adj" past de bevroren config toe (calibratie uit).
    """
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
    sel[~((sel != "") & (sterk >= C.MIN_STRENGTH))] = ""
    return sel


def uniek(df: pd.DataFrame, sel: np.ndarray) -> set:
    d = df.assign(_sel=sel)
    d = d[d["_sel"] != ""].sort_values("run_id")
    return set(zip(d["match_id"], d["_sel"]))


# =====================================================================
# DEEL 1: RECONSTRUCTIECONTROLE
# =====================================================================

def reconstructie(df: pd.DataFrame, ontbreekt: list[str]) -> tuple[str, np.ndarray | None]:
    hr("DEEL 1 - KAN DE SELECTIE EXACT WORDEN GERECONSTRUEERD?")
    print(f"  periode            {VAN} t/m {TOT} (wedstrijddatum)")
    print(f"  rijen              {len(df):,} (run x wedstrijd), {df['run_id'].nunique():,} runs, "
          f"{df['match_id'].nunique():,} wedstrijden")

    kritiek = [c for c in NODIG if c not in ("selection",)]
    if ontbreekt:
        print(f"\n  ONTBREKENDE KOLOMMEN: {ontbreekt}")
        if any(c in kritiek for c in ontbreekt):
            print("  -> reconstructie onmogelijk zonder deze kolommen.")
            return "ONMOGELIJK", None

    print("\n  [1] dekking (aandeel gevuld)")
    for c in ["odds_home", "prob_home", "bet_home", "rating_gap", "n_snapshots", "home_drift_pct",
              "rule_strength_adj_home", "rule_strength_calibrated_home"]:
        print(f"      {c:32s} {100 * df[c].notna().mean():6.1f}%")
    print("      (drift en strength mogen leeg zijn; de engine behandelt leeg als 0)")

    print("\n  [2] regelvlaggen opnieuw berekend tegen opgeslagen vlaggen")
    akkoord = []
    for side in ("home", "away"):
        her = herbereken_vlag(df, side)
        opg = df[f"{side}_rule"]
        gelijk = float((her == opg).mean())
        akkoord.append(gelijk)
        print(f"      {side:5s}: {100 * gelijk:7.3f}% gelijk  "
              f"(alleen herberekend: {int((her & ~opg).sum())}, alleen opgeslagen: {int((~her & opg).sum())})")
    vlag_ok = min(akkoord)

    print("\n  [3] destijds gemaakte MAIN-picks nabootsen met de destijds gebruikte strength")
    na = kies_main(df, "calibrated")
    echt = np.where(df["is_pick"].to_numpy(bool) & (df["pick_type"].to_numpy() == "MAIN"),
                    df["selection"].to_numpy(), "")
    tp = int(((na != "") & (na == echt)).sum())
    fp = int(((na != "") & (echt == "")).sum())
    fn = int(((na == "") & (echt != "")).sum())
    zw = int(((na != "") & (echt != "") & (na != echt)).sum())
    totaal = tp + fp + fn + zw
    na_ok = tp / totaal if totaal else 1.0
    print(f"      overeenstemming {100 * na_ok:7.3f}%  (gelijk {tp}, alleen nagebootst {fp}, "
          f"alleen destijds {fn}, andere kant {zw})")

    print("\n  [4] effect van het uitzetten van de strength-calibratie (bevroren config)")
    bev = kies_main(df, "adj")
    verschil_sterkte = (df["rule_strength_adj_home"].fillna(0) != df["rule_strength_calibrated_home"].fillna(0)) | \
                       (df["rule_strength_adj_away"].fillna(0) != df["rule_strength_calibrated_away"].fillna(0))
    regel = df["home_rule"] | df["away_rule"]
    print(f"      rijen waar de regel slaagt: {int(regel.sum()):,}; daarvan met "
          f"gecalibreerde strength <> ongecalibreerd: {int((verschil_sterkte & regel).sum()):,}")
    u_echt, u_bev = uniek(df, echt), uniek(df, bev)
    print(f"      unieke picks destijds {len(u_echt)}, met bevroren config {len(u_bev)}; "
          f"gemeen {len(u_echt & u_bev)}, alleen destijds {len(u_echt - u_bev)}, "
          f"alleen bevroren {len(u_bev - u_echt)}")

    if vlag_ok == 1.0 and na_ok == 1.0:
        oordeel = "EXACT"
    elif vlag_ok >= EXACT_DREMPEL and na_ok >= EXACT_DREMPEL:
        oordeel = "VRIJWEL EXACT"
    else:
        oordeel = "BENADERING"
    print(f"\n  >>> OORDEEL RECONSTRUCTIE: {oordeel}")
    if oordeel != "BENADERING":
        print("      De bevroren regels zijn op deze periode toe te passen met de opgeslagen")
        print("      waarden. Het enige verschil met destijds is de strength-calibratie (zie [4]).")
    return oordeel, bev


# =====================================================================
# DEEL 2: VERGELIJKING
# =====================================================================

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


def interval(m: float, se: float, G: int, schaal: float = 100.0) -> str:
    if not np.isfinite(se) or G < 2:
        return f"{schaal * m:+6.2f} (n.v.t.)"
    c = stats.t.ppf(0.975, G - 1)
    return f"{schaal * m:+6.2f} [{schaal * (m - c * se):+6.2f}; {schaal * (m + c * se):+6.2f}]"


def verschil(y1: np.ndarray, g1: np.ndarray, y2: np.ndarray, g2: np.ndarray) -> dict:
    """Verschil in gemiddelde (groep 1 - groep 2), cluster-robuust per competitie."""
    y = np.concatenate([y1, y2]).astype(float)
    d = np.concatenate([np.ones(len(y1)), np.zeros(len(y2))])
    gr = np.concatenate([g1, g2])
    X = np.column_stack([np.ones(len(y)), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    codes = pd.factorize(pd.Series(gr))[0]
    G = int(codes.max()) + 1
    meat = np.zeros((2, 2))
    for g in range(G):
        m = codes == g
        u = X[m].T @ r[m]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1))
    se = float(np.sqrt(V[1, 1]))
    return {"diff": float(b[1]), "se": se, "G": G, "t": float(b[1]) / se if se > 0 else np.nan}


def bouw_kandidaten(df: pd.DataFrame, bev: np.ndarray) -> pd.DataFrame:
    stukken = []
    for side, code in (("home", "HOME"), ("away", "AWAY")):
        stukken.append(pd.DataFrame({
            "run_id": df["run_id"].values,
            "match_id": df["match_id"].values,
            "competition": df["competition"].values,
            "date": df["date"].astype(str).str[:10].values,
            "kant": code,
            "odds": df[f"odds_{side}"].values,
            "o_h": df["odds_home"].values,
            "o_d": df["odds_draw"].values,
            "o_a": df["odds_away"].values,
            "uren_voor_aftrap": df["hours_to_kickoff"].values,
            "B": marktdeel(df, side).values,
            "A": (bev == code),
        }))
    return pd.concat(stukken, ignore_index=True)


def eerste(lang: pd.DataFrame, vlag: str) -> pd.DataFrame:
    s = lang[lang[vlag]].sort_values("run_id")
    return s.drop_duplicates(["match_id", "kant"], keep="first").copy()


def koppel_uitkomsten(e: pd.DataFrame, link: pd.DataFrame, closing: pd.DataFrame,
                      uitslag: pd.DataFrame) -> pd.DataFrame:
    e = e.copy()
    e["fixture_id"] = e["match_id"].map(link.set_index("match_id")["fixture_id"])
    e = e.merge(closing, on="fixture_id", how="left").merge(uitslag, on="fixture_id", how="left")
    e["p_close"] = np.where(e["kant"] == "HOME", e["pc_home"], e["pc_away"])
    gespeeld = e["status_short"].isin(["FT", "AET", "PEN"]) & e["ft_home_goals"].notna()
    hg, ag = e["ft_home_goals"], e["ft_away_goals"]
    e["won"] = np.where(e["kant"] == "HOME", hg > ag, ag > hg).astype(float)
    e["bruikbaar"] = gespeeld & e["p_close"].notna() & e["odds"].notna()
    e["clv"] = e["p_close"] * e["odds"] - 1.0
    e["uitslag"] = e["won"] - e["p_close"]
    e["roi"] = np.where(e["won"] == 1, e["odds"] - 1.0, -1.0)

    # Deel 3: CLV = marge bij instap + beweging tot de slotkoers.
    # marge_instap gebruikt alleen de prijs op het instapmoment (geen toekomst).
    ent = compute_market_probs(e[["o_h", "o_d", "o_a"]], odds_cols=("o_h", "o_d", "o_a"),
                               out_cols=("pe_home", "pe_draw", "pe_away"))
    e["p_instap"] = np.where(e["kant"] == "HOME", ent["pe_home"], ent["pe_away"])
    e["overround_instap"] = ent["market_overround"]
    e["marge_instap"] = e["p_instap"] * e["odds"] - 1.0
    e["beweging"] = e["clv"] - e["marge_instap"]          # = (p_close - p_instap) x odds
    return e


ODDSBANDEN = [1.50, 1.65, 1.80, 2.00, 2.5001]
BAND_LABELS = ["1.50-1.65", "1.65-1.80", "1.80-2.00", "2.00-2.50"]


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


def deel3(groepen: dict) -> None:
    hr("DEEL 3 - WAAR ZIT HET CLV-VERSCHIL?  CLV = marge bij instap + beweging")
    print("  Leesregel, vastgelegd voor de eerste run van dit deel:")
    print("  - Zit het verschil Beide - Alleen B vooral in de MARGE BIJ INSTAP, dan komt")
    print("    het door prijsniveau of competitiemix en zegt het niets over ECI.")
    print("  - Blijft het verschil in de BEWEGING overeind, ook gecorrigeerd voor oddsband,")
    print("    dan bewoog de markt na instap richting de ECI-selectie: selectiewaarde,")
    print("    met de onzekerheid die het interval aangeeft.")

    print("\n  PER GROEP  (gemiddelde [95%-interval], in %)")
    rijen = []
    for naam, g in groepen.items():
        ok = g[g["bruikbaar"] & g["p_instap"].notna()]
        cl = ok["competition"].to_numpy()
        r = {"groep": naam, "n": len(ok), "overround": round(ok["overround_instap"].mean(), 3)}
        for kol, label in (("clv", "CLV"), ("marge_instap", "marge bij instap"), ("beweging", "beweging")):
            m, se, G = cluster_mean(ok[kol].to_numpy(float), cl)
            r[label] = interval(m, se, G)
        rijen.append(r)
    print(pd.DataFrame(rijen).to_string(index=False))

    g1 = groepen["Beide (B-moment)"]
    g2 = groepen["Alleen B (B-moment)"]
    g1 = g1[g1["bruikbaar"] & g1["p_instap"].notna()]
    g2 = g2[g2["bruikbaar"] & g2["p_instap"].notna()]
    if len(g1) < 2 or len(g2) < 2:
        print("\n  Te weinig waarnemingen voor de verschillen.")
        return

    print("\n  BEIDE - ALLEEN B")
    rijen = []
    for kol, label in (("clv", "CLV"), ("marge_instap", "marge bij instap"), ("beweging", "beweging")):
        v = verschil(g1[kol].to_numpy(float), g1["competition"].to_numpy(),
                     g2[kol].to_numpy(float), g2["competition"].to_numpy())
        crit = stats.t.ppf(0.975, v["G"] - 1)
        rijen.append({"maat": label, "verschil": f"{100 * v['diff']:+.2f}",
                      "95%-interval": f"[{100 * (v['diff'] - crit * v['se']):+.2f}; "
                                      f"{100 * (v['diff'] + crit * v['se']):+.2f}]",
                      "t": round(v["t"], 2), "G": v["G"]})

    # Beweging, gecorrigeerd voor oddsband (dummies per band)
    samen = pd.concat([g1.assign(D=1.0), g2.assign(D=0.0)], ignore_index=True)
    samen["band"] = pd.cut(samen["odds"], ODDSBANDEN, labels=BAND_LABELS, right=False)
    samen = samen[samen["band"].notna()]
    dummies = pd.get_dummies(samen["band"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(samen)), samen["D"].to_numpy(float), dummies.to_numpy(float)])
    b, se, G = ols_cluster(samen["beweging"].to_numpy(float), X, samen["competition"].to_numpy())
    crit = stats.t.ppf(0.975, G - 1)
    rijen.append({"maat": "beweging, gecorrigeerd voor oddsband", "verschil": f"{100 * b[1]:+.2f}",
                  "95%-interval": f"[{100 * (b[1] - crit * se[1]):+.2f}; {100 * (b[1] + crit * se[1]):+.2f}]",
                  "t": round(b[1] / se[1], 2) if se[1] > 0 else np.nan, "G": G})
    print(pd.DataFrame(rijen).to_string(index=False))

    print("\n  BEWEGING PER ODDSBAND  (gemiddelde, in %)")
    rijen = []
    for band, s in samen.groupby("band", observed=False):
        a, z = s[s["D"] == 1], s[s["D"] == 0]
        rijen.append({"oddsband": band, "n Beide": len(a), "n Alleen B": len(z),
                      "beweging Beide": round(100 * a["beweging"].mean(), 2) if len(a) else np.nan,
                      "beweging Alleen B": round(100 * z["beweging"].mean(), 2) if len(z) else np.nan})
    print(pd.DataFrame(rijen).to_string(index=False))


def band_verschil(g1: pd.DataFrame, g2: pd.DataFrame, kol: str) -> dict:
    """Verschil g1 - g2 in kol, gecorrigeerd voor oddsband (zelfde model als deel 3)."""
    samen = pd.concat([g1.assign(D=1.0), g2.assign(D=0.0)], ignore_index=True)
    samen["band"] = pd.cut(samen["odds"], ODDSBANDEN, labels=BAND_LABELS, right=False)
    samen = samen[samen["band"].notna()]
    dummies = pd.get_dummies(samen["band"], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(samen)), samen["D"].to_numpy(float), dummies.to_numpy(float)])
    b, se, G = ols_cluster(samen[kol].to_numpy(float), X, samen["competition"].to_numpy())
    crit = stats.t.ppf(0.975, G - 1)
    return {"diff": b[1], "lo": b[1] - crit * se[1], "hi": b[1] + crit * se[1],
            "t": b[1] / se[1] if se[1] > 0 else np.nan, "G": G}


def controle_deel3(groepen: dict, B: pd.DataFrame) -> None:
    """Rekencontrole op deel 3. Verandert geen enkele uitkomst van deel 1-3."""
    hr("CONTROLE DEEL 3 - KLOPPEN DE BEREKENINGEN?")
    print("  Per kandidaat; Shin-kansen; odds = odds van de kant op het instapmoment:")
    print("    marge bij instap = p_instap x odds - 1   (p_instap: Bet365-odds die het model op de instaprun zag)")
    print("    CLV              = p_close  x odds - 1   (p_close: laatste Bet365-snapshot op of voor de aftrap)")
    print("    beweging         = CLV - marge bij instap, algebraisch gelijk aan (p_close - p_instap) x odds")

    alle = pd.concat([g.assign(_groep=n) for n, g in groepen.items()], ignore_index=True)
    ok = alle[alle["bruikbaar"] & alle["p_instap"].notna()]
    d_som = float((ok["clv"] - (ok["marge_instap"] + ok["beweging"])).abs().max())
    d_alg = float((ok["beweging"] - (ok["p_close"] - ok["p_instap"]) * ok["odds"]).abs().max())
    d_clv = float((ok["clv"] - (ok["p_close"] * ok["odds"] - 1.0)).abs().max())
    print(f"\n  Grootste afwijking per kandidaat, over {len(ok)} rijen in alle groepen:")
    print(f"    CLV - (marge + beweging)                 {d_som:.1e}")
    print(f"    beweging - (p_close - p_instap) x odds   {d_alg:.1e}")
    print(f"    CLV - (p_close x odds - 1)               {d_clv:.1e}")
    fout = max(d_som, d_alg, d_clv) > 1e-9
    print(f"    -> {'FOUT: afwijking groter dan rekenafronding' if fout else 'klopt (alleen rekenafronding, < 1e-9)'}")

    print("\n  Groepsgemiddelden onafgerond, in %:")
    print(f"    {'groep':22s} {'CLV':>9s} {'marge':>9s} {'beweging':>9s} {'CLV-(marge+bew.)':>17s}")
    for naam, g in groepen.items():
        s = g[g["bruikbaar"] & g["p_instap"].notna()]
        m = 100 * s[["clv", "marge_instap", "beweging"]].mean()
        rest = m["clv"] - m["marge_instap"] - m["beweging"]
        print(f"    {naam:22s} {m['clv']:+9.4f} {m['marge_instap']:+9.4f} {m['beweging']:+9.4f} {rest:+17.1e}")
    print("  (de tabellen hierboven ronden af op 2 decimalen; daardoor kan de som daar 0,01 afwijken)")

    beide = B[B["ook_A"] & B["bruikbaar"]]
    zelfde = int(beide["A"].sum())
    later = int((~beide["A"].astype(bool)).sum())
    print(f"\n  Instapmoment van Beide: A koos {zelfde} van de {len(beide)} kandidaten al op het")
    print(f"  B-instapmoment zelf, {later} pas in een latere run. Bij die {later} hangt het")
    print("  lidmaatschap van Beide deels af van wat er na het instapmoment gebeurde.")

    g1 = B[B["A"].astype(bool) & B["bruikbaar"] & B["p_instap"].notna()]
    g2 = B[~B["A"].astype(bool) & B["bruikbaar"] & B["p_instap"].notna()]
    if len(g1) >= 2 and len(g2) >= 2:
        r = band_verschil(g1, g2, "beweging")
        print(f"\n  Ter informatie, met de indeling op het instapmoment zelf (n = {len(g1)} tegen {len(g2)}):")
        print(f"    beweging, gecorrigeerd voor oddsband: {100 * r['diff']:+.2f} "
              f"[{100 * r['lo']:+.2f}; {100 * r['hi']:+.2f}], t = {r['t']:.2f}, G = {r['G']}")
        print("  Dit is beschrijvend en verandert de vooraf vastgelegde bevestigingstoets niet")
        print("  (zie PREREG_PRIJSBEWEGING.md).")


def vergelijking(df: pd.DataFrame, bev: np.ndarray, export: bool) -> None:
    hr("DEEL 2 - A (HUIDIGE REGELS) TEGENOVER B (ZONDER ECI-CRITERIA)")
    lang = bouw_kandidaten(df, bev)
    A, B = eerste(lang, "A"), eerste(lang, "B")
    sleutels_a = set(zip(A["match_id"], A["kant"]))
    sleutels_b = set(zip(B["match_id"], B["kant"]))
    B["ook_A"] = [k in sleutels_a for k in zip(B["match_id"], B["kant"])]
    gemeen = len(sleutels_a & sleutels_b)
    print(f"  A kiest {len(sleutels_a)} (wedstrijd, kant)-combinaties, B {len(sleutels_b)}; "
          f"gemeen {gemeen}")
    if len(sleutels_a - sleutels_b):
        print(f"  LET OP: {len(sleutels_a - sleutels_b)} A-picks zitten niet in B; dat hoort 0 te zijn.")

    link, _ = load_link()
    ids = sorted({int(i) for i in pd.concat([A, B])["match_id"]
                  .map(link.set_index("match_id")["fixture_id"]).dropna()})
    closing = build_closing(load_snapshots(ids), load_kickoffs(ids))
    if not closing.empty:
        kick = pd.to_datetime(closing["kickoff_at"], utc=True, errors="coerce")
        assert (kick.dt.date < FREEZE_START).all(), "er zit een wedstrijd uit de meetperiode in"
    closing = compute_market_probs(closing, odds_cols=("close_home", "close_draw", "close_away"),
                                   out_cols=("pc_home", "pc_draw", "pc_away"))
    closing = closing[["fixture_id", "pc_home", "pc_away"]]
    uitslag = load_results(ids)

    A = koppel_uitkomsten(A, link, closing, uitslag)
    B = koppel_uitkomsten(B, link, closing, uitslag)
    groepen = {
        "A (eigen moment)": A,
        "Beide (B-moment)": B[B["ook_A"]],
        "Alleen B (B-moment)": B[~B["ook_A"]],
        "B geheel (B-moment)": B,
    }
    weken = ((TOT - VAN).days + 1) / 7.0

    print("\n  AANTALLEN")
    rijen = []
    for naam, g in groepen.items():
        ok = g[g["bruikbaar"]]
        rijen.append({"groep": naam, "kandidaten": len(g), "bruikbaar": len(ok),
                      "wedstrijden": ok["match_id"].nunique(),
                      "gem_odds": round(ok["odds"].mean(), 2),
                      "uren_voor_aftrap": round(ok["uren_voor_aftrap"].median(), 1),
                      "per_week": round(len(ok) / weken, 1)})
    print(pd.DataFrame(rijen).to_string(index=False))
    print("  (bruikbaar = met slotkoers en uitslag; uren = mediaan op het instapmoment)")

    print("\n  UITKOMSTEN  (gemiddelde [95%-interval], in % of procentpunt)")
    rijen = []
    for naam, g in groepen.items():
        ok = g[g["bruikbaar"]]
        cl = ok["competition"].to_numpy()
        r = {"groep": naam, "n": len(ok)}
        for kol, label in (("clv", "CLV %"), ("uitslag", "uitslag-slotkans pp"), ("roi", "ROI %")):
            m, se, G = cluster_mean(ok[kol].to_numpy(float), cl)
            r[label] = interval(m, se, G)
        rijen.append(r)
    print(pd.DataFrame(rijen).to_string(index=False))

    print("\n  VERSCHILLEN  (cluster-robuust per competitie)")
    paren = [("Beide - Alleen B", groepen["Beide (B-moment)"], groepen["Alleen B (B-moment)"]),
             ("A - Alleen B", groepen["A (eigen moment)"], groepen["Alleen B (B-moment)"])]
    rijen = []
    for naam, g1, g2 in paren:
        g1, g2 = g1[g1["bruikbaar"]], g2[g2["bruikbaar"]]
        if len(g1) < 2 or len(g2) < 2:
            continue
        for kol, label in (("clv", "CLV"), ("uitslag", "uitslag-slotkans"), ("roi", "ROI")):
            v = verschil(g1[kol].to_numpy(float), g1["competition"].to_numpy(),
                         g2[kol].to_numpy(float), g2["competition"].to_numpy())
            crit = stats.t.ppf(0.975, v["G"] - 1)
            rijen.append({"vergelijking": naam, "maat": label,
                          "verschil": f"{100 * v['diff']:+.2f}",
                          "95%-interval": f"[{100 * (v['diff'] - crit * v['se']):+.2f}; "
                                          f"{100 * (v['diff'] + crit * v['se']):+.2f}]",
                          "t": round(v["t"], 2), "G": v["G"]})
    print(pd.DataFrame(rijen).to_string(index=False))

    # Timing: dezelfde kandidaten, A-moment tegen B-moment (alleen CLV zinvol)
    a_ok = A[A["bruikbaar"]].set_index(["match_id", "kant"])
    b_ok = B[B["bruikbaar"] & B["ook_A"]].set_index(["match_id", "kant"])
    paar = a_ok[["clv", "competition"]].join(b_ok[["clv"]], rsuffix="_b", how="inner")
    if len(paar) >= 2:
        m, se, G = cluster_mean((paar["clv"] - paar["clv_b"]).to_numpy(float),
                                paar["competition"].to_numpy())
        print(f"\n  A - Beide, zelfde {len(paar)} kandidaten, ander moment: CLV {interval(m, se, G)}")
        print("  (positief = wachten op het ECI-signaal leverde een betere prijs op)")

    deel3(groepen)
    controle_deel3(groepen, B)

    print("\n  Lezen: Beide - Alleen B is de kernvraag. Verschilt die niet van nul, dan")
    print("  sluiten de ECI-filters geen aantoonbaar slechtere kandidaten uit en is A")
    print("  in de praktijk een willekeurige uitdunning van B. Per bet geldt:")
    print("  ROI = CLV + (uitslag - slotkans) x odds.")

    if export:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        p = EXPORT_DIR / f"selectie_vergelijking_{stamp}.csv"
        pd.concat([A.assign(methode="A"), B.assign(methode="B")]).to_csv(p, index=False)
        print(f"\n[export] {p}")


# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="A (huidige regels) tegen B (zonder ECI), juli-augustus 2026")
    ap.add_argument("--forceer", action="store_true",
                    help="vergelijk ook als de reconstructie geen (vrijwel) exacte match is")
    ap.add_argument("--export-csv", action="store_true")
    args = ap.parse_args()

    hr("CONFIG")
    controleer_config()

    ruw, ontbreekt = load_snapshot_rows()
    df = prepare(ruw)
    dubbel = int(df.duplicated(["run_id", "match_id"]).sum())
    df = df.drop_duplicates(["run_id", "match_id"]).reset_index(drop=True)
    datums = pd.to_datetime(df["date"].astype(str).str[:10], errors="coerce").dt.date
    assert (datums < FREEZE_START).all(), "er zit een wedstrijd uit de meetperiode in"
    if dubbel:
        print(f"  dubbele (run, wedstrijd)-rijen verwijderd: {dubbel}")

    oordeel, bev = reconstructie(df, ontbreekt)
    if oordeel in ("ONMOGELIJK",) or (oordeel == "BENADERING" and not args.forceer):
        print("\nGestopt: de selectie is niet (vrijwel) exact te reconstrueren. Kijk eerst naar")
        print("deel 1. Met --forceer draait de vergelijking toch, maar dan als benadering.")
        return
    if oordeel == "BENADERING":
        print("\n!! Vergelijking draait als BENADERING (--forceer). Lees de uitkomst zo.")

    vergelijking(df, bev, args.export_csv)
    print("\nKlaar. Read-only: er is niets gewijzigd.")


if __name__ == "__main__":
    main()
