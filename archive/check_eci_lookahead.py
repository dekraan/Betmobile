"""
check_eci_lookahead.py
======================

Toetst of de ratings in eci_data historisch zijn (rating zoals die gold vóór
de wedstrijd) of besmet met informatie van ná de wedstrijd.

De feed levert per wedstrijd twee waarden:
    ECI            bevroren op het moment van de wedstrijd
    RankingPoints  de huidige ranglijstwaarde

Door de opgeslagen home_rating te vergelijken met beide, zien we welke van de
twee de scraper heeft binnengehaald. Sluit hij aan op ECI, dan is je historie
schoon. Sluit hij aan op RankingPoints, dan zit er lookahead in alles wat na
afloop is opgehaald of bijgewerkt.

READ-ONLY. Draait alleen SELECT-queries.
Gebruikt DB_CONFIG uit betmobile_settings.py, net als je andere scripts.

Gebruik
-------
    python check_eci_lookahead.py
    python check_eci_lookahead.py --run eci_snapshots\\run_20260831T1358Z
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 ontbreekt. Installeer met: pip install psycopg2-binary")

try:
    from betmobile_settings import DB_CONFIG
except ImportError:
    sys.exit(
        "betmobile_settings.py niet gevonden.\n"
        "Draai dit script vanuit je Betmobile-map, of zet het daarin neer."
    )


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def norm(name: str) -> str:
    """Namen vergelijkbaar maken: accenten weg, kleine letters, geen spaties."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(s.lower().split())


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- deel 1: wanneer is er opgehaald ten opzichte van de wedstrijd ----------

def timing_analysis(cur) -> None:
    hr("DEEL 1 - opgehaald voor of na de wedstrijd?")

    cur.execute("""
        WITH d AS (
            SELECT
                CASE WHEN date ~ '^\\d{4}-\\d{2}-\\d{2}$'
                     THEN date::date END AS match_date,
                created_at, updated_at
            FROM eci_data
        )
        SELECT
            count(*)                                                      AS totaal,
            count(*) FILTER (WHERE match_date IS NULL)                    AS geen_datum,
            count(*) FILTER (WHERE created_at::date <= match_date)        AS aangemaakt_voor,
            count(*) FILTER (WHERE created_at::date >  match_date)        AS aangemaakt_na,
            count(*) FILTER (WHERE updated_at::date >  match_date)        AS bijgewerkt_na,
            count(*) FILTER (WHERE updated_at IS DISTINCT FROM created_at
                             AND updated_at::date > match_date)           AS echt_gewijzigd_na
        FROM d
    """)
    row = cur.fetchone()
    labels = [
        ("rijen totaal", row[0]),
        ("datum onbruikbaar", row[1]),
        ("aangemaakt op of voor wedstrijddag", row[2]),
        ("aangemaakt NA de wedstrijd", row[3]),
        ("bijgewerkt NA de wedstrijd", row[4]),
        ("bijgewerkt na, en gewijzigd t.o.v. aanmaak", row[5]),
    ]
    for label, value in labels:
        pct = f"{100.0 * value / row[0]:5.1f}%" if row[0] else "    -"
        print(f"  {label:<44} {value:>8,}  {pct}")

    print("\n  Rijen die NA de wedstrijd zijn aangemaakt of bijgewerkt lopen")
    print("  risico op een rating die de uitslag al kent. Deel 2 toetst of")
    print("  dat risico zich ook werkelijk voordoet.")


# --- deel 2: vergelijken met de feed ---------------------------------------

def load_feed(run_dir: Path) -> dict:
    path = run_dir / "matches.jsonl"
    if not path.exists():
        sys.exit(f"Geen matches.jsonl in {run_dir}")

    feed = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d = str(r.get("d_Date", ""))[:10]
            key = (d, norm(r.get("c_HomeTeam")), norm(r.get("c_Awayteam")))
            feed[key] = r
    return feed


def compare_with_feed(cur, feed: dict) -> None:
    hr("DEEL 2 - opgeslagen rating vergeleken met de feed")

    dates = sorted({k[0] for k in feed})
    if not dates:
        sys.exit("Feed is leeg.")
    print(f"  feed: {len(feed):,} wedstrijden, {dates[0]} tot {dates[-1]}")

    cur.execute("""
        SELECT date, home_team, away_team, home_rating, away_rating,
               created_at, updated_at
        FROM eci_data
        WHERE date >= %s AND date <= %s
          AND home_rating IS NOT NULL
    """, (dates[0], dates[-1]))
    rows = cur.fetchall()
    print(f"  eci_data in datzelfde bereik: {len(rows):,} rijen")

    matched, unmatched = [], []
    for date_s, home, away, hr_val, ar_val, created, updated in rows:
        key = (str(date_s)[:10], norm(home), norm(away))
        rec = feed.get(key)
        if rec is None:
            unmatched.append((date_s, home, away))
            continue
        matched.append((rec, hr_val, ar_val, created, updated))

    print(f"  gekoppeld op datum + teamnamen: {len(matched):,}")
    print(f"  niet gekoppeld: {len(unmatched):,}")
    if unmatched[:5]:
        print("    voorbeelden van niet-gekoppelde rijen:")
        for d, h, a in unmatched[:5]:
            print(f"      {d}  {h} - {a}")

    if not matched:
        print("\n  Geen overlap. Zonder koppeling kunnen we niets concluderen.")
        return

    # Voor elke rij: ligt de opgeslagen waarde dichter bij ECI of bij RankingPoints?
    buckets = {"afgespeeld": [], "nog te spelen": []}

    for rec, hr_val, ar_val, created, updated in matched:
        played = str(rec.get("hasResults")) == "1"
        bucket = "afgespeeld" if played else "nog te spelen"

        for stored, eci_key, rank_key in (
            (hr_val, "HomeECI", "HomeRankingPoints"),
            (ar_val, "AwayECI", "AwayRankingPoints"),
        ):
            s = to_float(stored)
            e = to_float(rec.get(eci_key))
            r = to_float(rec.get(rank_key))
            if s is None or e is None or r is None:
                continue
            buckets[bucket].append((abs(s - e), abs(s - r)))

    for label, pairs in buckets.items():
        print(f"\n  --- {label}: {len(pairs):,} rating-waarnemingen ---")
        if not pairs:
            continue

        # Binnen 0.5 betekent: dit is exact deze waarde, afgerond.
        near_eci = sum(1 for de, dr in pairs if de <= 0.5)
        near_rank = sum(1 for de, dr in pairs if dr <= 0.5)
        closer_eci = sum(1 for de, dr in pairs if de < dr)
        n = len(pairs)

        print(f"    komt overeen met ECI (binnen 0.5)           "
              f"{near_eci:>7,} / {n:,}  ({100.0*near_eci/n:5.1f}%)")
        print(f"    komt overeen met RankingPoints (binnen 0.5) "
              f"{near_rank:>7,} / {n:,}  ({100.0*near_rank/n:5.1f}%)")
        print(f"    dichter bij ECI dan bij RankingPoints       "
              f"{closer_eci:>7,} / {n:,}  ({100.0*closer_eci/n:5.1f}%)")

        mean_e = sum(de for de, _ in pairs) / n
        mean_r = sum(dr for _, dr in pairs) / n
        print(f"    gemiddeld verschil met ECI: {mean_e:8.2f}")
        print(f"    gemiddeld verschil met RankingPoints: {mean_r:8.2f}")

    print("\n  HOE TE LEZEN (kijk naar de afgespeelde wedstrijden)")
    print("    hoog percentage bij ECI            -> historie is schoon,")
    print("                                          bruikbaar voor backtests")
    print("    hoog percentage bij RankingPoints  -> lookahead: de opgeslagen")
    print("                                          rating kent de uitslag al")
    print("    geen van beide                     -> de scraper haalt iets")
    print("                                          anders op; nader kijken")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toets op lookahead in eci_data")
    parser.add_argument("--run", help="pad naar een run-map met matches.jsonl")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        root = Path("eci_snapshots")
        runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
        if not runs:
            sys.exit("Geen eci_snapshots gevonden. Geef een map op met --run")
        run_dir = runs[-1]

    print(f"check_eci_lookahead.py   {datetime.now():%Y-%m-%d %H:%M}")
    print(f"run: {run_dir}")
    print(f"database: {DB_CONFIG.get('database', DB_CONFIG.get('dbname'))} "
          f"op {DB_CONFIG.get('host')}:{DB_CONFIG.get('port')} "
          f"als {DB_CONFIG.get('user')}")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        sys.exit(f"[--] verbinden mislukt: {exc}")

    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()

    timing_analysis(cur)
    compare_with_feed(cur, load_feed(run_dir))

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()