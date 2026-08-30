"""
collect_t1_horizons.py

VERZAMELSCRIPT VOOR HYPOTHESE T1 — geen analyse.

T1: voorspelt het prijsverschil tussen Pinnacle en Bet365 op moment T de
latere beweging van Bet365 richting de slotkoers?

Dit script verzamelt alleen de meetpunten. Het rekent niets uit, toetst
niets en trekt geen conclusies. Dat gebeurt pas bij de toets zelf, en pas
op data vanaf de freeze date (30-08-2026).

MEETREGEL
    Voor horizon H geldt: de LAATSTE snapshot op of vóór kickoff - H uur.
    Dus de prijs die je op dat moment daadwerkelijk gezien zou hebben,
    niet de dichtstbijzijnde in beide richtingen. Voor een timingvraag is
    dat de juiste semantiek — je kunt niet vooruitkijken.

    De werkelijke afstand tot het doelmoment wordt vastgelegd als
    *_offset_h, zodat bij de analyse gefilterd kan worden op verse
    meetpunten.

HORIZONS
    48, 24, 12, 6 en 3 uur vóór aftrap. Vastgelegd op 30-08-2026, vóór
    enige meting. Niet uitbreiden of aanpassen zonder dat expliciet te
    noteren — anders kan achteraf de gunstigste horizon gekozen worden.

Gebruik:
    python collect_t1_horizons.py                    # vanaf de freeze date
    python collect_t1_horizons.py --since 2026-04-01 # ruimere periode
    python collect_t1_horizons.py --dry-run          # alleen dekking tonen
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import db_engine
from prob_calibration import compute_market_probs

# ---------------------------------------------------------------------
# Vaste parameters — vastgelegd 30-08-2026
# ---------------------------------------------------------------------

HORIZONS = [48, 24, 12, 6, 3]

PINNACLE_ID = 4
BET365_ID = 8

FREEZE_DATE = "2026-08-30"

TABEL = "t1_market_horizons"


DDL = f"""
CREATE TABLE IF NOT EXISTS {TABEL} (
    fixture_id          integer     NOT NULL,
    kickoff_at          timestamptz NOT NULL,
    horizon_h           integer     NOT NULL,

    pin_captured_at     timestamptz,
    pin_offset_h        numeric,
    pin_p_home          numeric,
    pin_p_draw          numeric,
    pin_p_away          numeric,

    b365_captured_at    timestamptz,
    b365_offset_h       numeric,
    b365_p_home         numeric,
    b365_p_draw         numeric,
    b365_p_away         numeric,

    close_captured_at   timestamptz,
    close_offset_h      numeric,
    close_p_home        numeric,
    close_p_draw        numeric,
    close_p_away        numeric,

    collected_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (fixture_id, horizon_h)
);
"""


def controleer_labels(engine) -> dict[str, str]:
    """Zoek uit hoe de 1x2-labels heten; faal hard als ze onbekend zijn."""
    q = text("""
        SELECT DISTINCT label FROM odds_values_snapshots
        WHERE market_key = '1x2' LIMIT 20
    """)
    with engine.connect() as c:
        labels = [r[0] for r in c.execute(q)]

    kaart = {}
    for lab in labels:
        k = lab.strip().lower()
        if k in ("home", "1", "h"):
            kaart["home"] = lab
        elif k in ("draw", "x", "d"):
            kaart["draw"] = lab
        elif k in ("away", "2", "a"):
            kaart["away"] = lab

    ontbreekt = {"home", "draw", "away"} - set(kaart)
    if ontbreekt:
        raise RuntimeError(
            f"1x2-labels niet herkend. Gevonden: {labels}. "
            f"Ontbreekt: {sorted(ontbreekt)}. Pas controleer_labels() aan."
        )
    return kaart


SQL_HAAL = """
WITH doel AS (
    SELECT
        f.fixture_id,
        f.date_utc AS kickoff,
        h.horizon
    FROM fixtures f
    CROSS JOIN (VALUES {horizon_values}) AS h(horizon)
    WHERE f.date_utc >= :since
      AND f.date_utc <  now()
      AND f.status_short = 'FT'
      AND NOT EXISTS (
          SELECT 1 FROM {tabel} t
          WHERE t.fixture_id = f.fixture_id AND t.horizon_h = h.horizon
      )
),
snap AS (
    SELECT
        d.fixture_id,
        d.kickoff,
        d.horizon,
        bm.bookmaker_id,
        s.captured_at,
        s.p_home, s.p_draw, s.p_away
    FROM doel d
    CROSS JOIN (VALUES ({pin}),({b365})) AS bm(bookmaker_id)
    LEFT JOIN LATERAL (
        SELECT
            o.captured_at,
            MAX(o.odd) FILTER (WHERE o.label = :lab_home) AS p_home,
            MAX(o.odd) FILTER (WHERE o.label = :lab_draw) AS p_draw,
            MAX(o.odd) FILTER (WHERE o.label = :lab_away) AS p_away
        FROM odds_values_snapshots o
        WHERE o.fixture_id   = d.fixture_id
          AND o.bookmaker_id = bm.bookmaker_id
          AND o.market_key   = '1x2'
          AND o.captured_at <= d.kickoff - make_interval(hours => d.horizon)
        GROUP BY o.captured_at
        HAVING COUNT(*) FILTER (
            WHERE o.label IN (:lab_home, :lab_draw, :lab_away)
        ) = 3
        ORDER BY o.captured_at DESC
        LIMIT 1
    ) s ON true
),
slot AS (
    SELECT DISTINCT ON (d.fixture_id)
        d.fixture_id,
        s.captured_at AS close_captured_at,
        s.p_home AS close_home, s.p_draw AS close_draw, s.p_away AS close_away
    FROM (SELECT DISTINCT fixture_id, kickoff FROM doel) d
    LEFT JOIN LATERAL (
        SELECT
            o.captured_at,
            MAX(o.odd) FILTER (WHERE o.label = :lab_home) AS p_home,
            MAX(o.odd) FILTER (WHERE o.label = :lab_draw) AS p_draw,
            MAX(o.odd) FILTER (WHERE o.label = :lab_away) AS p_away
        FROM odds_values_snapshots o
        WHERE o.fixture_id   = d.fixture_id
          AND o.bookmaker_id = {b365}
          AND o.market_key   = '1x2'
          AND o.captured_at <  d.kickoff
        GROUP BY o.captured_at
        HAVING COUNT(*) FILTER (
            WHERE o.label IN (:lab_home, :lab_draw, :lab_away)
        ) = 3
        ORDER BY o.captured_at DESC
        LIMIT 1
    ) s ON true
    ORDER BY d.fixture_id
)
SELECT
    sn.fixture_id,
    sn.kickoff,
    sn.horizon,
    sn.bookmaker_id,
    sn.captured_at,
    EXTRACT(EPOCH FROM (sn.kickoff - sn.captured_at)) / 3600.0 AS offset_h,
    sn.p_home AS odd_home, sn.p_draw AS odd_draw, sn.p_away AS odd_away,
    sl.close_captured_at,
    EXTRACT(EPOCH FROM (sn.kickoff - sl.close_captured_at)) / 3600.0 AS close_offset_h,
    sl.close_home, sl.close_draw, sl.close_away
FROM snap sn
LEFT JOIN slot sl ON sl.fixture_id = sn.fixture_id
WHERE sn.captured_at IS NOT NULL
"""


def devig(df: pd.DataFrame, cols: tuple[str, str, str], uit: tuple[str, str, str]) -> pd.DataFrame:
    """Devig 1x2-odds naar kansen volgens Shin. Rijen met NaN blijven NaN."""
    ok = df[list(cols)].notna().all(axis=1) & (df[list(cols)] > 1.0).all(axis=1)
    for c in uit:
        df[c] = np.nan
    if ok.any():
        sub = compute_market_probs(
            df.loc[ok, list(cols)].copy(), odds_cols=cols, out_cols=uit, method="shin"
        )
        for c in uit:
            df.loc[ok, c] = sub[c].values
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=FREEZE_DATE,
                    help=f"Vanaf welke kickoff-datum (default: freeze {FREEZE_DATE})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Alleen dekking tonen, niets wegschrijven")
    args = ap.parse_args()

    engine = db_engine()

    if args.since != FREEZE_DATE:
        print(f"!! LET OP: --since {args.since} ligt vóór de freeze date {FREEZE_DATE}.")
        print("!! Deze data telt NIET mee voor de toets van T1; alleen voor dekkingsinspectie.")

    with engine.begin() as c:
        c.execute(text(DDL))

    labels = controleer_labels(engine)
    print(f"[labels] home={labels['home']!r} draw={labels['draw']!r} away={labels['away']!r}")

    sql = text(
        SQL_HAAL.format(
            tabel=TABEL,
            horizon_values=",".join(f"({h})" for h in HORIZONS),
            pin=PINNACLE_ID,
            b365=BET365_ID,
        )
    ).bindparams(
        since=args.since,
        lab_home=labels["home"],
        lab_draw=labels["draw"],
        lab_away=labels["away"],
    )

    df = pd.read_sql(sql, engine)
    if df.empty:
        print("[info] geen nieuwe fixtures om te verwerken.")
        return

    print(f"[haal] {len(df)} snapshotrijen over "
          f"{df['fixture_id'].nunique()} fixtures")

    # Devigging
    df = devig(df, ("odd_home", "odd_draw", "odd_away"), ("p_home", "p_draw", "p_away"))
    df = devig(df, ("close_home", "close_draw", "close_away"),
               ("cp_home", "cp_draw", "cp_away"))

    # Van lang naar breed: één rij per (fixture, horizon)
    pin = df[df.bookmaker_id == PINNACLE_ID].set_index(["fixture_id", "horizon"])
    b3 = df[df.bookmaker_id == BET365_ID].set_index(["fixture_id", "horizon"])

    uit = pd.DataFrame(index=pin.index.union(b3.index)).reset_index()
    uit = uit.merge(
        pin[["kickoff", "captured_at", "offset_h", "p_home", "p_draw", "p_away"]]
        .rename(columns={"captured_at": "pin_captured_at", "offset_h": "pin_offset_h",
                         "p_home": "pin_p_home", "p_draw": "pin_p_draw",
                         "p_away": "pin_p_away"}),
        on=["fixture_id", "horizon"], how="left",
    ).merge(
        b3[["captured_at", "offset_h", "p_home", "p_draw", "p_away",
            "close_captured_at", "close_offset_h", "cp_home", "cp_draw", "cp_away"]]
        .rename(columns={"captured_at": "b365_captured_at", "offset_h": "b365_offset_h",
                         "p_home": "b365_p_home", "p_draw": "b365_p_draw",
                         "p_away": "b365_p_away", "cp_home": "close_p_home",
                         "cp_draw": "close_p_draw", "cp_away": "close_p_away"}),
        on=["fixture_id", "horizon"], how="left",
    ).rename(columns={"horizon": "horizon_h", "kickoff": "kickoff_at"})

    uit = uit[uit["kickoff_at"].notna()]

    # Dekkingsrapport
    print("\n=== DEKKING PER HORIZON ===")
    rap = uit.groupby("horizon_h").agg(
        fixtures=("fixture_id", "nunique"),
        pin_ok=("pin_p_home", lambda s: s.notna().sum()),
        b365_ok=("b365_p_home", lambda s: s.notna().sum()),
        beide=("fixture_id", "size"),
        close_ok=("close_p_home", lambda s: s.notna().sum()),
        gem_pin_offset=("pin_offset_h", "mean"),
        gem_b365_offset=("b365_offset_h", "mean"),
    ).round(2)
    rap["compleet"] = uit.groupby("horizon_h").apply(
        lambda g: (g[["pin_p_home", "b365_p_home", "close_p_home"]]
                   .notna().all(axis=1)).sum()
    )
    print(rap.to_string())

    if args.dry_run:
        print("\n[dry-run] niets weggeschreven.")
        return

    kol = ["fixture_id", "kickoff_at", "horizon_h",
           "pin_captured_at", "pin_offset_h", "pin_p_home", "pin_p_draw", "pin_p_away",
           "b365_captured_at", "b365_offset_h", "b365_p_home", "b365_p_draw", "b365_p_away",
           "close_captured_at", "close_offset_h", "close_p_home", "close_p_draw",
           "close_p_away"]
    uit[kol].to_sql(TABEL, engine, if_exists="append", index=False)
    print(f"\n[schrijf] {len(uit)} rijen naar {TABEL}")

    with engine.connect() as c:
        n = c.execute(text(f"SELECT COUNT(DISTINCT fixture_id) FROM {TABEL}")).scalar()
    print(f"[totaal] {n} fixtures in {TABEL}")


if __name__ == "__main__":
    main()