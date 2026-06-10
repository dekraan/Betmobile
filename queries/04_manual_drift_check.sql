-- =========================================================
-- 04_manual_drift_check.sql
-- Doel:
-- Handmatig drift berekenen (eerste vs laatste odds)
-- Dit is je "waarheid"
-- =========================================================

WITH base AS (
    SELECT
        fixture_id,
        label,
        MIN(captured_at) AS first_time,
        MAX(captured_at) AS last_time
    FROM odds_values_snapshots
    WHERE bookmaker_id = 4
      AND market_key = '1x2'
    GROUP BY fixture_id, label
),
first_last AS (
    SELECT
        b.fixture_id,
        b.label,
        f.odd AS first_odd,
        l.odd AS last_odd
    FROM base b
    JOIN odds_values_snapshots f
        ON f.fixture_id = b.fixture_id
        AND f.label = b.label
        AND f.captured_at = b.first_time
        AND f.bookmaker_id = 4
        AND f.market_key = '1x2'
    JOIN odds_values_snapshots l
        ON l.fixture_id = b.fixture_id
        AND l.label = b.label
        AND l.captured_at = b.last_time
        AND l.bookmaker_id = 4
        AND l.market_key = '1x2'
)
SELECT
    fixture_id,
    label,
    first_odd,
    last_odd,
    ROUND(((last_odd - first_odd) / first_odd)::numeric, 4) AS drift_pct
FROM first_last
WHERE fixture_id = 1378164;