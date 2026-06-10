-- =========================================================
-- 07_find_big_drift.sql
-- Doel:
-- Wedstrijden met duidelijke marktbeweging
-- =========================================================

SELECT
    fixture_id,
    home_drift_pct,
    away_drift_pct,
    n_snapshots,
    hours_to_kickoff
FROM public.odds_1x2_bet365_dynamics_now_mv
WHERE ABS(home_drift_pct) > 0.05
   OR ABS(away_drift_pct) > 0.05
ORDER BY GREATEST(ABS(home_drift_pct), ABS(away_drift_pct)) DESC;