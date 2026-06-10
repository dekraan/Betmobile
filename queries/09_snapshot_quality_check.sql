-- =========================================================
-- 09_snapshot_quality_check.sql
-- Doel:
-- Check of je genoeg data hebt voor drift
-- =========================================================

SELECT
    fixture_id,
    n_snapshots,
    home_range,
    away_range,
    hours_stale
FROM public.odds_1x2_bet365_dynamics_now_mv
ORDER BY n_snapshots ASC
LIMIT 50;