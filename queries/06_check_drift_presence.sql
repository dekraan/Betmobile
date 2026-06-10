-- =========================================================
-- 06_check_drift_presence.sql
-- Doel:
-- Snel zien of drift ergens voorkomt
-- =========================================================

SELECT
    COUNT(*) AS total_matches,
    COUNT(*) FILTER (WHERE ABS(home_drift_pct) > 0.01) AS home_drift,
    COUNT(*) FILTER (WHERE ABS(away_drift_pct) > 0.01) AS away_drift
FROM public.odds_1x2_bet365_dynamics_now_mv;