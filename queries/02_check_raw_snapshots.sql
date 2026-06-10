-- =========================================================
-- 02_check_raw_snapshots.sql
-- Doel:
-- Laat alle ruwe odds snapshots zien voor 1 wedstrijd
-- Gebruik om te zien of odds echt bewegen
-- =========================================================

SELECT
    fixture_id,
    label,
    odd,
    captured_at
FROM odds_values_snapshots
WHERE fixture_id = 1378164  -- aanpassen
  AND bookmaker_id = 4
ORDER BY captured_at;