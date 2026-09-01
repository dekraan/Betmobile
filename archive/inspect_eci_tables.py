"""
inspect_eci_tables.py
=====================

Brengt in kaart wat er in je ECI-tabellen staat, zodat we kunnen bepalen of
de opgeslagen ratings historisch zijn (rating ten tijde van de wedstrijd) of
huidig (rating op het moment van ophalen).

Waarom dit ertoe doet
---------------------
De feed blijkt twee ratings te hebben: ECI (bevroren per wedstrijd) en
RankingPoints (huidige waarde). De oude scraper leest de gerenderde pagina.
Als daar de huidige waarde in stond voor al afgespeelde wedstrijden, dan zit
er in eerdere backtests informatie van NA de wedstrijd. Dat moeten we weten
voordat we verder bouwen.

Dit script is READ-ONLY. Het draait alleen SELECT-queries en schrijft niets.

Gebruik
-------
    pip install psycopg2-binary

    set PGPASSWORD=...
    python inspect_eci_tables.py

Of vul de CONFIG hieronder in.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

# --- CONFIG -----------------------------------------------------------------
# Alles is te overrulen met argumenten op de commandoregel.
# Zet het wachtwoord NIET in dit bestand; gebruik PGPASSWORD of de prompt.

from betmobile_settings import DB_CONFIG

# Tabellen en views waar we naar kijken. Aanvullen mag.
TARGETS = [
    "eci_data",
    "eci_data_snapshots",
    "eci_fixture_links_v",
]

SAMPLE_ROWS = 3

# ----------------------------------------------------------------------------

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("psycopg2 ontbreekt. Installeer met: pip install psycopg2-binary")


def connect():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(
            f"[OK] verbonden als '{DB_CONFIG['user']}' "
            f"met database '{DB_CONFIG['database']}'"
        )
        return conn
    except psycopg2.OperationalError as exc:
        print(f"[--] verbinden mislukt: {exc}")
        sys.exit(1)


def q(cur, sql, params=None):
    cur.execute(sql, params or ())
    return cur.fetchall()


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------

def list_eci_objects(cur) -> None:
    hr("ALLE OBJECTEN MET 'eci' IN DE NAAM")
    rows = q(cur, """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_name ILIKE '%%eci%%'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
    """)
    if not rows:
        print("  geen gevonden")
        return
    for schema, name, ttype in rows:
        print(f"  {schema}.{name:<40} {ttype}")

    # Materialized views staan niet in information_schema.tables.
    rows = q(cur, """
        SELECT schemaname, matviewname
        FROM pg_matviews
        WHERE matviewname ILIKE '%%eci%%'
        ORDER BY 1, 2
    """)
    for schema, name in rows:
        print(f"  {schema}.{name:<40} MATERIALIZED VIEW")


def describe(cur, table: str) -> list[tuple[str, str]]:
    hr(f"TABEL: {table}")

    cols = q(cur, """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """, (table,))

    if not cols:
        print("  bestaat niet (of geen rechten)")
        return []

    print(f"  {len(cols)} kolommen:")
    for name, dtype, nullable in cols:
        null = "NULL" if nullable == "YES" else "    "
        print(f"    {name:<32} {dtype:<28} {null}")

    try:
        n = q(cur, f'SELECT count(*) FROM "{table}"')[0][0]
        print(f"\n  rijen: {n:,}")
    except psycopg2.Error as exc:
        print(f"\n  [--] tellen mislukt: {exc}")
        cur.connection.rollback()
        return cols

    return cols


def date_ranges(cur, table: str, cols) -> None:
    """Bereik van alle datum- en tijdkolommen. Dit onthult of er naast een
    wedstrijddatum ook een ophaalmoment wordt bewaard."""
    date_cols = [
        c[0] for c in cols
        if any(k in c[1] for k in ("date", "timestamp"))
    ]
    if not date_cols:
        print("  geen datum/tijd-kolommen")
        return

    print("\n  datum- en tijdkolommen:")
    for col in date_cols:
        try:
            row = q(cur, f'''
                SELECT min("{col}"), max("{col}"), count("{col}")
                FROM "{table}"
            ''')[0]
            print(f"    {col:<32} {str(row[0])[:19]}  tot  {str(row[1])[:19]}"
                  f"   ({row[2]:,} gevuld)")
        except psycopg2.Error as exc:
            cur.connection.rollback()
            print(f"    {col:<32} [--] {str(exc).strip()[:60]}")


def rating_columns(cur, table: str, cols) -> None:
    """Kolommen die op een rating lijken, met hun spreiding. Als een team
    steeds dezelfde waarde heeft, is het een huidige waarde en geen
    historische."""
    candidates = [
        c[0] for c in cols
        if any(k in c[0].lower() for k in ("eci", "rating", "rank", "points"))
        and any(k in c[1] for k in ("numeric", "double", "real", "integer", "bigint"))
    ]
    if not candidates:
        print("\n  geen kolommen die op een rating lijken")
        return

    print("\n  rating-achtige kolommen:")
    for col in candidates:
        try:
            row = q(cur, f'''
                SELECT count(*), count(DISTINCT "{col}"),
                       min("{col}"), max("{col}"), avg("{col}")
                FROM "{table}"
                WHERE "{col}" IS NOT NULL
            ''')[0]
            avg = f"{row[4]:.1f}" if row[4] is not None else "-"
            print(f"    {col:<32} n={row[0]:<8,} uniek={row[1]:<8,} "
                  f"min={row[2]} max={row[3]} gem={avg}")
        except psycopg2.Error as exc:
            cur.connection.rollback()
            print(f"    {col:<32} [--] {str(exc).strip()[:60]}")


def sample(cur, table: str) -> None:
    print(f"\n  {SAMPLE_ROWS} voorbeeldrijen:")
    try:
        cur.execute(f'SELECT * FROM "{table}" LIMIT {SAMPLE_ROWS}')
        rows = cur.fetchall()
        names = [d[0] for d in cur.description]
    except psycopg2.Error as exc:
        cur.connection.rollback()
        print(f"    [--] {exc}")
        return

    for i, row in enumerate(rows, start=1):
        print(f"\n    --- rij {i} ---")
        for name, value in zip(names, row):
            text = str(value)
            if len(text) > 58:
                text = text[:55] + "..."
            print(f"      {name:<30} {text}")


def decimal_check(cur, table: str, cols) -> None:
    """Zijn de opgeslagen ratings afgerond op hele punten? De oude scraper
    las de gerenderde pagina, en die rondt af met Math.round(). Als alle
    waarden heel zijn, komt de data uit de HTML en niet uit de feed."""
    candidates = [
        c[0] for c in cols
        if any(k in c[0].lower() for k in ("eci", "rating"))
        and any(k in c[1] for k in ("numeric", "double", "real"))
    ]
    if not candidates:
        return

    print("\n  afronding (heel getal = uit de gerenderde pagina):")
    for col in candidates:
        try:
            row = q(cur, f'''
                SELECT count(*) AS n,
                       count(*) FILTER (WHERE "{col}" = round("{col}")) AS heel
                FROM "{table}"
                WHERE "{col}" IS NOT NULL
            ''')[0]
            n, heel = row[0], row[1]
            if n:
                pct = 100.0 * heel / n
                verdict = "ALLES afgerond" if pct > 99.5 else "decimalen aanwezig"
                print(f"    {col:<32} {heel:,}/{n:,} heel ({pct:.1f}%)  -> {verdict}")
        except psycopg2.Error as exc:
            cur.connection.rollback()
            print(f"    {col:<32} [--] {str(exc).strip()[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecteer de ECI-tabellen (read-only)")
    parser.add_argument("--tables", help="komma-gescheiden tabelnamen (overschrijft TARGETS)")
    args = parser.parse_args()

    targets = [t.strip() for t in args.tables.split(",")] if args.tables else TARGETS

    print(f"inspect_eci_tables.py   {date.today()}")
    print(
        f"database: {DB_CONFIG['database']} "
        f"op {DB_CONFIG['host']}:{DB_CONFIG['port']} "
        f"als {DB_CONFIG['user']}\n"
)
    conn = connect()
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()

    list_eci_objects(cur)

    for table in targets:
        cols = describe(cur, table)
        if not cols:
            continue
        date_ranges(cur, table, cols)
        rating_columns(cur, table, cols)
        decimal_check(cur, table, cols)
        sample(cur, table)

    cur.close()
    conn.close()

    hr("KLAAR")
    print("Stuur de uitvoer door. Op basis van de kolomnamen bouwen we de")
    print("vergelijking tussen wat er in de database staat en wat de feed geeft.")


if __name__ == "__main__":
    main()