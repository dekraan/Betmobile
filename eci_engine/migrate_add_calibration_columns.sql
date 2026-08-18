-- Migratie: probability-calibratie kolommen in picks_evaluated.
-- Eenmalig draaien in PGAdmin (Query Tool) VOOR de eerste run met
-- USE_PROB_CALIBRATION = True.
--
-- prob_home/draw/away bevatten na de omschakeling de GEKALIBREERDE kansen;
-- de rauwe ECI-kansen staan vanaf dan in prob_*_raw. Oude rijen houden
-- NULL in de nieuwe kolommen — zo zie je voor altijd precies welke picks
-- onder welk regime gegenereerd zijn.

ALTER TABLE public.picks_evaluated
    ADD COLUMN IF NOT EXISTS prob_home_raw       double precision,
    ADD COLUMN IF NOT EXISTS prob_draw_raw       double precision,
    ADD COLUMN IF NOT EXISTS prob_away_raw       double precision,
    ADD COLUMN IF NOT EXISTS calibration_class   text,
    ADD COLUMN IF NOT EXISTS calibration_w       double precision,
    ADD COLUMN IF NOT EXISTS calibration_version text;

-- Controle:
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'picks_evaluated' AND column_name LIKE '%calib%';
