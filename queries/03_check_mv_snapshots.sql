-- =========================================================
-- 03_check_mv_snapshots.sql
-- Doel:
-- Controleer of snapshots MV correct gevuld is
-- (na refresh!)
-- =========================================================

SELECT *
FROM public.odds_1x2_bet365_snapshots_mv
WHERE fixture_id = 1378164
ORDER BY captured_at;