-- Migratie: ontdubbelde picks-view.
-- Een weddenschap = de EERSTE keer dat een (match_id, selection) als pick
-- verscheen; latere runs die dezelfde pick herhalen tellen niet mee.
-- Eenmalig draaien in PGAdmin.
--
-- LET OP DISTINCT ON: de DISTINCT-kolommen moeten vooraan in ORDER BY staan.

CREATE OR REPLACE VIEW public.picks_evaluated_unique_v AS
SELECT DISTINCT ON (match_id, selection) *
FROM public.picks_evaluated
ORDER BY match_id, selection, run_id ASC;

-- Controle: unieke bets horen ~241 te zijn (jouw eerdere query):
-- SELECT COUNT(*) FROM public.picks_evaluated_unique_v WHERE outcome IN ('WIN','LOSS');
