-- =========================================================
-- 01_refresh_all_views.sql
-- Doel:
-- Forceer actuele data in alle materialized views
-- Gebruik dit vóór elke analyse van drift / odds / picks
-- =========================================================

REFRESH MATERIALIZED VIEW public.odds_1x2_bet365_snapshots_mv;
REFRESH MATERIALIZED VIEW public.odds_1x2_bet365_latest_now_mv;
REFRESH MATERIALIZED VIEW public.odds_1x2_bet365_first_seen_mv;
REFRESH MATERIALIZED VIEW public.odds_1x2_bet365_agg_now_mv;
REFRESH MATERIALIZED VIEW public.odds_1x2_bet365_dynamics_now_mv;

REFRESH MATERIALIZED VIEW public.eci_fixture_link_mv;
REFRESH MATERIALIZED VIEW public.betmobile_api_mv;
REFRESH MATERIALIZED VIEW public.betmobile_api_ready_mv;