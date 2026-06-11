# settle_picks.py
# Vult score/result/outcome/settled_at voor picks_evaluated en picks_single_fail_candidates
# op basis van public.eci_data.eci_score.

import psycopg2
from datetime import datetime
from pathlib import Path

from betmobile_settings import DB_CONFIG

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "settle_picks_log.txt"


SQL_SETTLE_PICKS_EVALUATED = """
UPDATE public.picks_evaluated p
SET
    score = e.eci_score,
    result = CASE
        WHEN split_part(e.eci_score, '-', 1)::int > split_part(e.eci_score, '-', 2)::int THEN 'HOME'
        WHEN split_part(e.eci_score, '-', 1)::int < split_part(e.eci_score, '-', 2)::int THEN 'AWAY'
        ELSE 'DRAW'
    END,
    outcome = CASE
        WHEN p.selection = CASE
            WHEN split_part(e.eci_score, '-', 1)::int > split_part(e.eci_score, '-', 2)::int THEN 'HOME'
            WHEN split_part(e.eci_score, '-', 1)::int < split_part(e.eci_score, '-', 2)::int THEN 'AWAY'
            ELSE 'DRAW'
        END THEN 'WIN'
        ELSE 'LOSS'
    END,
    settled_at = NOW(),
    date_ts = COALESCE(p.date_ts, p.date::timestamptz)
FROM public.eci_data e
WHERE p.match_id = e.match_id
  AND p.outcome IS NULL
  AND e.eci_score ~ '^[0-9]+-[0-9]+$';
"""

SQL_SETTLE_SINGLE_FAILS = """
UPDATE public.picks_single_fail_candidates sf
SET
    score = e.eci_score,
    result = CASE
        WHEN split_part(e.eci_score, '-', 1)::int > split_part(e.eci_score, '-', 2)::int THEN 'HOME'
        WHEN split_part(e.eci_score, '-', 1)::int < split_part(e.eci_score, '-', 2)::int THEN 'AWAY'
        ELSE 'DRAW'
    END,
    outcome = CASE
        WHEN sf.side = CASE
            WHEN split_part(e.eci_score, '-', 1)::int > split_part(e.eci_score, '-', 2)::int THEN 'HOME'
            WHEN split_part(e.eci_score, '-', 1)::int < split_part(e.eci_score, '-', 2)::int THEN 'AWAY'
            ELSE 'DRAW'
        END THEN 'WIN'
        ELSE 'LOSS'
    END,
    settled_at = NOW()
FROM public.eci_data e
WHERE sf.match_id = e.match_id
  AND sf.outcome IS NULL
  AND e.eci_score ~ '^[0-9]+-[0-9]+$';
"""

SQL_SETTLE_NEAR_MISSES = """
UPDATE public.picks_near_miss_candidates nm
SET
    score = e.eci_score,
    result = CASE
        WHEN split_part(e.eci_score, '-', 1)::int > split_part(e.eci_score, '-', 2)::int THEN 'HOME'
        WHEN split_part(e.eci_score, '-', 1)::int < split_part(e.eci_score, '-', 2)::int THEN 'AWAY'
        ELSE 'DRAW'
    END,
    outcome = CASE
        WHEN nm.side = CASE
            WHEN split_part(e.eci_score, '-', 1)::int > split_part(e.eci_score, '-', 2)::int THEN 'HOME'
            WHEN split_part(e.eci_score, '-', 1)::int < split_part(e.eci_score, '-', 2)::int THEN 'AWAY'
            ELSE 'DRAW'
        END THEN 'WIN'
        ELSE 'LOSS'
    END,
    settled_at = NOW()
FROM public.eci_data e
WHERE nm.match_id = e.match_id
  AND nm.outcome IS NULL
  AND e.eci_score ~ '^[0-9]+-[0-9]+$';
"""

SQL_SUMMARY = """
SELECT
    'picks_evaluated' AS table_name,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE outcome IS NOT NULL) AS settled,
    COUNT(*) FILTER (WHERE outcome IS NULL) AS open
FROM public.picks_evaluated

UNION ALL

SELECT
    'picks_single_fail_candidates',
    COUNT(*),
    COUNT(*) FILTER (WHERE outcome IS NOT NULL),
    COUNT(*) FILTER (WHERE outcome IS NULL)
FROM public.picks_single_fail_candidates

UNION ALL

SELECT
    'picks_near_miss_candidates',
    COUNT(*),
    COUNT(*) FILTER (WHERE outcome IS NOT NULL),
    COUNT(*) FILTER (WHERE outcome IS NULL)
FROM public.picks_near_miss_candidates;
"""


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    log("SETTLE PICKS START")

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_SETTLE_PICKS_EVALUATED)
                updated_picks = cur.rowcount

                cur.execute(SQL_SETTLE_SINGLE_FAILS)
                updated_single_fails = cur.rowcount

                cur.execute(SQL_SETTLE_NEAR_MISSES)
                updated_near_misses = cur.rowcount

                conn.commit()

                log(f"Updated picks_evaluated: {updated_picks}")
                log(f"Updated picks_single_fail_candidates: {updated_single_fails}")
                log(f"Updated picks_near_miss_candidates: {updated_near_misses}")

                cur.execute(SQL_SUMMARY)
                rows = cur.fetchall()

                for table_name, total, settled, open_count in rows:
                    log(
                        f"{table_name}: total={total}, settled={settled}, open={open_count}"
                    )

        log("SETTLE PICKS DONE")

    except Exception as e:
        log(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    main()