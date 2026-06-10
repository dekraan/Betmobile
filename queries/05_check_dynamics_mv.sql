-- =========================================================
-- 05_check_dynamics_mv.sql
-- Doel:
-- Controleer berekende drift en features
-- =========================================================

SELECT
    fixture_id,
    home_drift_pct,
    away_drift_pct,
    home_last_move_pct,
    away_last_move_pct,
    home_recent24_pct,
    away_recent24_pct,
    n_snapshots,
    home_range,
    away_range,
    hours_stale
FROM public.odds_1x2_bet365_dynamics_now_mv
WHERE fixture_id = 1378164;