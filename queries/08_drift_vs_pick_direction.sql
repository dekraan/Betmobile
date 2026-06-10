-- =========================================================
-- 08_drift_vs_pick_direction.sql
-- Doel:
-- Zie of jouw pick meegaat met of tegen de markt
-- =========================================================

SELECT
    match_id,
    home_team,
    away_team,
    odds_home,
    odds_away,
    home_win_pct,
    away_win_pct,
    home_drift_pct,
    away_drift_pct,
    CASE
        WHEN away_win_pct > home_win_pct AND away_drift_pct <= -0.03 THEN 'AWAY + SUPPORT'
        WHEN away_win_pct > home_win_pct AND away_drift_pct >= 0.03 THEN 'AWAY + OPPOSE'
        WHEN home_win_pct > away_win_pct AND home_drift_pct <= -0.03 THEN 'HOME + SUPPORT'
        WHEN home_win_pct > away_win_pct AND home_drift_pct >= 0.03 THEN 'HOME + OPPOSE'
        ELSE 'MIXED'
    END AS drift_context
FROM public.betmobile_api_ready_mv
WHERE score IS NULL
ORDER BY date;