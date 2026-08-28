"""
clv_report.py

Closing Line Value (CLV) rapport voor Betmobile.

De vraag die dit script beantwoordt: verslaan onze picks de closing line?
Voor elke gesettelde pick vergelijken we de odds waartegen de pick is
opgenomen met de laatste Bet365-snapshot voor de aftrap (closing-proxy),
in ge-devigde kansruimte.

Waarom dit de beslissende test is: de calibratie-analyse liet zien dat
ECI-kansen geen informatie bevatten die de (closing) markt mist. Als het
systeem toch echte edge heeft, moet die uit timing komen - vroege prijzen
pakken die daarna de goede kant op bewegen. Dat is exact wat CLV meet,
en het convergeert veel sneller dan winst/verlies-resultaten.

Kernmetriek: edge_vs_close = p_close(selectie) x odds_genomen - 1
  > 0  : de closing markt vindt jouw genomen prijs een goede deal
  ~ 0  : geen timing-edge; winst/verlies is dan vooral variantie
  < 0  : je pakt structureel slechtere prijzen dan de close

Gebruik:
    python clv_report.py
    python clv_report.py --export-csv
    python clv_report.py --keep-all      # geen dedup naar eerste pick per match
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import OUTPUT_DIR
from db import db_engine, relation_exists

from prob_calibration import assign_competition_class, compute_market_probs
from shared_buckets import ODDS_BINS_REPORT, ODDS_LABELS_REPORT
from fit_calibration import CLASS_PATTERNS, DEFAULT_CLASS, _patterns_calib_stub

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_SOURCE = "odds_1x2_bet365_snapshots_mv"
FIXTURES_TABLE = "fixtures"
LINK_CANDIDATES = ["eci_fixture_link_mv", "eci_fixture_links_v", "eci_fixture_link_v"]


# =====================================================================
# HELPERS
# =====================================================================

def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_table(title: str, df: pd.DataFrame, max_rows: int = 60) -> None:
    print_header(title)
    if df is None or df.empty:
        print("Geen data.")
        return
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].round(4)
    print(view.to_string(index=False))


def get_table_columns(source: str, schema: str = "public") -> set[str]:
    q = text(
        """
        SELECT a.attname
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :source
          AND a.attnum > 0 AND NOT a.attisdropped
        """
    )
    with db_engine().connect() as conn:
        rows = conn.execute(q, {"schema": schema, "source": source}).fetchall()
    return {r[0] for r in rows}


# =====================================================================
# DATA LADEN
# =====================================================================

PICK_WANTED = [
    "run_id", "match_id", "competition", "date", "date_ts",
    "home_team", "away_team",
    "odds_home", "odds_draw", "odds_away",
    "selection", "outcome", "pick_type", "pick_tier",
    "rule_strength_adj", "n_snapshots",
    "home_drift_pct", "away_drift_pct",
]


def load_picks() -> pd.DataFrame:
    available = get_table_columns("picks_evaluated")
    cols = [c for c in PICK_WANTED if c in available]
    required = {"match_id", "selection", "outcome", "odds_home", "odds_draw", "odds_away"}
    missing = required - set(cols)
    if missing:
        raise RuntimeError(f"picks_evaluated mist kolommen: {sorted(missing)}")

    q = f"""
        SELECT {', '.join(cols)}
        FROM public.picks_evaluated
        WHERE outcome IN ('WIN', 'LOSS')
          AND selection IN ('HOME', 'DRAW', 'AWAY')
          AND rule_passed IS TRUE
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    print(f"[load] picks_evaluated: {len(df)} gesettelde picks")
    return df


def load_run_times() -> pd.DataFrame | None:
    """created_at per run, als picks_run bestaat (voor het pick-moment)."""
    exists, _ = relation_exists("picks_run")
    if not exists:
        return None
    available = get_table_columns("picks_run")
    id_col = next((c for c in ("run_id", "id") if c in available), None)
    if id_col is None or "created_at" not in available:
        return None
    with db_engine().connect() as conn:
        df = pd.read_sql(
            f"SELECT {id_col} AS run_id, created_at AS pick_created_at FROM public.picks_run",
            conn,
        )
    return df


def load_link() -> tuple[pd.DataFrame, str]:
    """ECI match_id -> Bet365 fixture_id."""
    for name in LINK_CANDIDATES:
        exists, _ = relation_exists(name)
        if not exists:
            continue
        available = get_table_columns(name)
        eci_col = next((c for c in ("match_id", "eci_match_id") if c in available), None)
        fx_col = "fixture_id" if "fixture_id" in available else None
        if eci_col and fx_col:
            with db_engine().connect() as conn:
                df = pd.read_sql(
                    f"SELECT {eci_col} AS match_id, {fx_col} AS fixture_id FROM public.{name}",
                    conn,
                )
            df = df.dropna().drop_duplicates(subset=["match_id"])
            print(f"[load] link via {name}: {len(df)} koppelingen")
            return df, name
    raise RuntimeError(
        f"Geen linkview gevonden (geprobeerd: {LINK_CANDIDATES}) "
        "of match_id/fixture_id kolommen ontbreken."
    )


def _ids_clause(ids: list[int]) -> str:
    safe = ",".join(str(int(i)) for i in ids)
    return safe if safe else "NULL"


def load_kickoffs(fixture_ids: list[int]) -> pd.DataFrame:
    q = f"""
        SELECT fixture_id, date_utc AS kickoff_at
        FROM public.{FIXTURES_TABLE}
        WHERE fixture_id IN ({_ids_clause(fixture_ids)})
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    df["kickoff_at"] = pd.to_datetime(df["kickoff_at"], utc=True, errors="coerce")
    return df


def load_snapshots(fixture_ids: list[int]) -> pd.DataFrame:
    q = f"""
        SELECT fixture_id, captured_at, odds_home, odds_draw, odds_away
        FROM public.{SNAPSHOT_SOURCE}
        WHERE fixture_id IN ({_ids_clause(fixture_ids)})
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True, errors="coerce")
    print(f"[load] snapshots: {len(df)} rijen voor {df['fixture_id'].nunique()} fixtures")
    return df


# =====================================================================
# KERN: CLOSING BEPALEN EN CLV BEREKENEN
# =====================================================================

def build_closing(snaps: pd.DataFrame, kickoffs: pd.DataFrame) -> pd.DataFrame:
    """Laatste snapshot VOOR de aftrap per fixture = closing-proxy."""
    df = snaps.merge(kickoffs, on="fixture_id", how="inner")
    df = df[df["captured_at"] <= df["kickoff_at"]]
    if df.empty:
        return pd.DataFrame(
            columns=["fixture_id", "kickoff_at", "close_captured_at",
                     "close_home", "close_draw", "close_away"]
        )
    df = df.sort_values("captured_at").groupby("fixture_id", as_index=False).last()
    df = df.rename(
        columns={
            "captured_at": "close_captured_at",
            "odds_home": "close_home",
            "odds_draw": "close_draw",
            "odds_away": "close_away",
        }
    )
    return df[["fixture_id", "kickoff_at", "close_captured_at",
               "close_home", "close_draw", "close_away"]]


def _selected(df: pd.DataFrame, home_col: str, draw_col: str, away_col: str) -> pd.Series:
    return np.select(
        [df["selection"] == "HOME", df["selection"] == "DRAW", df["selection"] == "AWAY"],
        [df[home_col], df[draw_col], df[away_col]],
        default=np.nan,
    )


def build_clv_frame(
    picks: pd.DataFrame,
    link: pd.DataFrame,
    closing: pd.DataFrame,
    run_times: pd.DataFrame | None,
    dedupe: bool = True,
) -> tuple[pd.DataFrame, dict]:
    coverage: dict[str, int] = {"picks_settled": len(picks)}
    df = picks.copy()

    if run_times is not None and "run_id" in df.columns:
        df = df.merge(run_times, on="run_id", how="left")
    else:
        df["pick_created_at"] = pd.NaT
    df["pick_created_at"] = pd.to_datetime(df["pick_created_at"], utc=True, errors="coerce")

    # Dedup: dezelfde match kan in meerdere runs als pick verschijnen.
    # De EERSTE keer is het echte inzetmoment; latere herhalingen zouden
    # de CLV-statistiek dubbel laten meetellen.
    if dedupe:
        before = len(df)
        sort_col = "pick_created_at" if df["pick_created_at"].notna().any() else "run_id"
        df = (
            df.sort_values(sort_col)
            .drop_duplicates(subset=["match_id", "selection"], keep="first")
        )
        coverage["na_dedup"] = len(df)
        if before != len(df):
            print(f"[dedup] {before} -> {len(df)} picks (eerste pick per match+selectie)")

    df = df.merge(link, on="match_id", how="left")
    coverage["met_fixture_link"] = int(df["fixture_id"].notna().sum())

    df = df.merge(closing, on="fixture_id", how="left")
    coverage["met_closing_snapshot"] = int(df["close_home"].notna().sum())

    # Closing moet NA het pick-moment liggen, anders meet je niets.
    known_created = df["pick_created_at"].notna() & df["close_captured_at"].notna()
    df["close_after_pick"] = np.where(
        known_created, df["close_captured_at"] > df["pick_created_at"], True
    )
    usable = df["close_home"].notna() & df["close_after_pick"]
    coverage["bruikbaar_voor_clv"] = int(usable.sum())

    df = df[usable].copy()
    if df.empty:
        return df, coverage

    # Ge-devigde kansen: genomen prijs en closing prijs.
    df = compute_market_probs(
        df,
        odds_cols=("odds_home", "odds_draw", "odds_away"),
        out_cols=("p_taken_home", "p_taken_draw", "p_taken_away"),
    )
    df = compute_market_probs(
        df,
        odds_cols=("close_home", "close_draw", "close_away"),
        out_cols=("p_close_home", "p_close_draw", "p_close_away"),
    )

    df["odds_taken_sel"] = _selected(df, "odds_home", "odds_draw", "odds_away")
    df["odds_close_sel"] = _selected(df, "close_home", "close_draw", "close_away")
    df["p_taken_sel"] = _selected(df, "p_taken_home", "p_taken_draw", "p_taken_away")
    df["p_close_sel"] = _selected(df, "p_close_home", "p_close_draw", "p_close_away")

    # De drie CLV-metrieken.
    df["clv_odds_pct"] = df["odds_taken_sel"] / df["odds_close_sel"] - 1.0
    df["prob_move"] = df["p_close_sel"] - df["p_taken_sel"]
    df["edge_vs_close"] = df["p_close_sel"] * df["odds_taken_sel"] - 1.0

    # Referentie: gerealiseerde winst van dezelfde picks.
    df["profit"] = np.where(df["outcome"] == "WIN", df["odds_taken_sel"] - 1.0, -1.0)

    # Context voor uitsplitsingen.
    stub = _patterns_calib_stub(CLASS_PATTERNS, DEFAULT_CLASS)
    df["competition_class"] = assign_competition_class(df["competition"], stub)

    df["selected_drift_pct"] = np.select(
        [df["selection"] == "HOME", df["selection"] == "AWAY"],
        [df.get("home_drift_pct", np.nan), df.get("away_drift_pct", np.nan)],
        default=np.nan,
    )
    df["drift_bucket"] = pd.cut(
        pd.to_numeric(df["selected_drift_pct"], errors="coerce"),
        bins=[-np.inf, -0.05, -0.02, 0.0, 0.02, np.inf],
        labels=["<=-5% (sterk mee)", "-5..-2%", "-2..0%", "0..+2%", ">+2% (tegen)"],
    )
    df["odds_bucket"] = pd.cut(
        pd.to_numeric(df["odds_taken_sel"], errors="coerce"),
        bins=ODDS_BINS_REPORT, labels=ODDS_LABELS_REPORT,
    )
    if "date_ts" in df.columns:
        df["month"] = pd.to_datetime(df["date_ts"], errors="coerce").dt.strftime("%Y-%m")

    df["hours_pick_to_close"] = (
        (df["close_captured_at"] - df["pick_created_at"]).dt.total_seconds() / 3600.0
    )

    return df, coverage


# =====================================================================
# RAPPORTAGE
# =====================================================================

def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    group = df.groupby(by, observed=True) if by else [("ALL", df)]
    rows = []
    iterator = group if by is None else group
    for name, part in iterator:
        n = len(part)
        if n == 0:
            continue
        edge = part["edge_vs_close"].to_numpy(dtype=float)
        se = edge.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append(
            {
                (by or "groep"): name,
                "picks": n,
                "clv_odds_pct": part["clv_odds_pct"].mean(),
                "share_clv_pos": (part["clv_odds_pct"] > 0).mean(),
                "edge_vs_close": edge.mean(),
                "t_stat": edge.mean() / se if se and se > 0 else np.nan,
                "realized_roi": part["profit"].mean(),
            }
        )
    return pd.DataFrame(rows)


def run_report(dedupe: bool = True, export_csv: bool = False) -> pd.DataFrame:
    print_header("BETMOBILE CLV RAPPORT")

    picks = load_picks()
    if picks.empty:
        print("Geen gesettelde picks gevonden.")
        return pd.DataFrame()

    link, _ = load_link()
    run_times = load_run_times()

    matched = picks.merge(link, on="match_id", how="inner")
    fixture_ids = sorted(set(int(x) for x in matched["fixture_id"].dropna()))
    if not fixture_ids:
        print("Geen enkele pick kon aan een fixture gekoppeld worden.")
        return pd.DataFrame()

    kickoffs = load_kickoffs(fixture_ids)
    snaps = load_snapshots(fixture_ids)
    closing = build_closing(snaps, kickoffs)

    df, coverage = build_clv_frame(picks, link, closing, run_times, dedupe=dedupe)

    print_header("DEKKING")
    for k, v in coverage.items():
        print(f"{k:>24}: {v}")
    if df.empty:
        print("Geen bruikbare picks voor CLV.")
        return df

    med_gap = df["hours_pick_to_close"].median()
    if pd.notna(med_gap):
        print(f"{'mediane uren pick->close':>24}: {med_gap:.1f}")

    print_table("CLV TOTAAL", summarize(df))
    print_table("CLV PER PICK TYPE", summarize(df, "pick_type"))
    if "pick_tier" in df.columns:
        print_table("CLV PER TIER", summarize(df, "pick_tier"))
    print_table("CLV PER DRIFT BUCKET (drift op pickmoment)", summarize(df, "drift_bucket"))
    print_table("CLV PER ODDS BUCKET", summarize(df, "odds_bucket"))
    print_table("CLV PER COMPETITIEKLASSE", summarize(df, "competition_class"))
    if "month" in df.columns:
        print_table("CLV PER MAAND", summarize(df, "month"))

    print_header("HOE TE LEZEN")
    print(
        "clv_odds_pct  = hoeveel beter je genomen prijs was dan de closing prijs.\n"
        "edge_vs_close = verwachte winst per unit ALS de ge-devigde closing kans\n"
        "                de waarheid is. Dit is de geldmetriek: hij moet ook de\n"
        "                bookmakermarge overwinnen, dus clv_odds_pct kan positief\n"
        "                zijn terwijl edge_vs_close nog rond 0 hangt.\n"
        "edge_vs_close > 0 met t_stat > 2 = hard bewijs van timing-edge.\n"
        "Rond 0 of negatief = de gerealiseerde ROI is waarschijnlijk variantie.\n"
        "Vergelijk vooral de drift-buckets: als de edge echt is, hoort hij daar\n"
        "het sterkst zichtbaar te zijn (drift mee = hogere CLV)."
    )

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail = EXPORT_DIR / f"clv_picks_detail_{stamp}.csv"
        df.to_csv(detail, index=False, encoding="utf-8-sig")
        print(f"\n[export] {detail}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Closing Line Value rapport")
    parser.add_argument("--keep-all", action="store_true",
                        help="Geen dedup naar eerste pick per match+selectie")
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()
    run_report(dedupe=not args.keep_all, export_csv=args.export_csv)


if __name__ == "__main__":
    main()