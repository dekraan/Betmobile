import psycopg2
from sqlalchemy import create_engine
from config import DB_DSN, DB_CONFIG


def db_engine():
    return create_engine(DB_DSN)

def db_conn():
    return psycopg2.connect(**DB_CONFIG)

def relation_exists(name: str) -> tuple[bool, str | None]:
    """
    Geeft terug:
    - (True, 'materialized_view') als het een materialized view is
    - (True, 'view') als het een gewone view is
    - (False, None) als het niet bestaat
    """
    q = """
        SELECT EXISTS (
            SELECT 1
            FROM pg_matviews
            WHERE schemaname = 'public'
              AND matviewname = %s
        ) AS is_matview,
        EXISTS (
            SELECT 1
            FROM information_schema.views
            WHERE table_schema = 'public'
              AND table_name = %s
        ) AS is_view
    """
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (name, name))
        is_matview, is_view = cur.fetchone()

    if is_matview:
        return True, "materialized_view"
    if is_view:
        return True, "view"
    return False, None

def auto_exclude_past_no_score_matches():
    sql = """
        UPDATE public.eci_data
        SET
            is_excluded = true,
            review_status = 'auto_excluded_no_score_after_date',
            review_flag = false,
            review_reason = 'past_date_no_score',
            flagged_at = CURRENT_TIMESTAMP
        WHERE
            eci_score IS NULL
            AND NULLIF(date, '') IS NOT NULL
            AND date::date < CURRENT_DATE - INTERVAL '2 days'
            AND COALESCE(is_excluded, false) = false;
    """

    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        affected = cur.rowcount
        conn.commit()

    print(f"[cleanup] oude ECI-wedstrijden zonder score uitgesloten: {affected}")
    return affected

def refresh_source_views():
    """
    Ververst alle onderliggende materialized views in de juiste volgorde.
    Gewone views hoef je niet te refreshen.
    """
    print("[views] controleren en refreshen...")

    candidates = [
        "odds_1x2_bet365_snapshots_mv",
        "odds_1x2_bet365_latest_now_mv",
        "odds_1x2_bet365_first_seen_mv",
        "odds_1x2_bet365_agg_now_mv",
        "odds_1x2_bet365_dynamics_now_mv",
        "eci_fixture_link_mv",
        "betmobile_api_mv",
        "betmobile_api_ready_mv",
    ]

    with db_conn() as conn, conn.cursor() as cur:
        for name in candidates:
            exists, kind = relation_exists(name)

            if not exists:
                print(f"[views][WARN] {name} bestaat niet.")
                continue

            if kind == "materialized_view":
                print(f"[views] REFRESH MATERIALIZED VIEW public.{name}")
                cur.execute(f"REFRESH MATERIALIZED VIEW public.{name};")
            else:
                print(f"[views] public.{name} is een gewone view, refresh niet nodig.")

        conn.commit()

    print("[views] klaar.")


def get_upcoming_source_name() -> str:
    """
    Gebruik bij voorkeur betmobile_api_ready_mv.
    Als die niet bestaat, val terug op betmobile_api_mv.
    """
    for name in ["betmobile_api_ready_mv", "betmobile_api_mv"]:
        exists, _kind = relation_exists(name)
        if exists:
            return name

    raise RuntimeError(
        "Geen bronview gevonden. Verwachtte public.betmobile_api_ready_mv of public.betmobile_api_mv."
    )