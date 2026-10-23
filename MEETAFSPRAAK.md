# Meetafspraak: out-of-sample toets van de bevroren pickconfiguratie

Vastgelegd op 23-10-2026, voordat de meetperiode is geëvalueerd. Dit document verduidelijkt de bestaande afspraak ("oordeel bij 150-200 picks, op ROI en edge vs close samen") en verandert niets aan de regels.

## Wat wordt getoetst

De pickconfiguratie zoals bevroren op 18-08-2026:

- `eci_engine/config.py`, laatst gewijzigd op 18-08-2026 (git `974219b`): kans ≥ 0,58, value ≥ 1,02, rating gap ≥ 500, odds 1,50–2,50, minstens 7 snapshots, drift niet tegen (≤ +3%), strength ≥ 1,8, secundaire picks aan, geen kans- of strength-calibratie.
- `rules.py` en `picks.py` zijn ongewijzigd.

Tot het oordeel wordt hier niets aan veranderd.

## Wat meetelt

- MAIN-picks (`pick_type = 'MAIN'`) uit `picks_evaluated_unique_v`. Per combinatie van wedstrijd en kant telt de eerste keer dat die als pick verscheen, tegen de odds van die run.
- Alleen wedstrijden op of na 18-08-2026.
- Alleen picks die zijn afgerekend als WIN of LOSS.
- Secundaire picks tellen niet mee voor het oordeel. Ze worden apart gerapporteerd, met dezelfde maten.

## Wanneer

Het oordeel valt bij **150 afgerekende MAIN-picks**. Niet eerder en niet later.

## Maten

- **Edge vs close (CLV):** `p_close × odds_genomen − 1`. Daarin is `p_close` de Shin-gedevigde kans uit de laatste Bet365-snapshot op of vóór de aftrap (`odds_1x2_bet365_snapshots_mv`), zoals in `clv_report.py`.
- **ROI** tegen de genomen odds.
- Beide met een 95%-interval, cluster-robuust per competitie, kritieke waarde t(G−1).
- Het oordeel valt op ROI en edge vs close samen, zoals afgesproken.

## Vooraf bekende kanttekeningen

- De Bet365-snapshots komen ongeveer elke twee uur binnen. De "slotkoers" ligt daardoor gemiddeld 1,5 à 2 uur voor de aftrap.
- Bij 150 picks is het 95%-interval op ROI ruwweg ±15 procentpunt. ROI kan dan alleen grote verschillen aantonen; CLV is veel preciezer.

## Scheiding

- De meetperiode wordt niet gebruikt om regels te ontwikkelen of bij te stellen.
- Analyses van de periode vóór 18-08-2026, zoals `research_selectie_vergelijking.py` over 1 juli t/m 17 augustus, zijn verkennend en veranderen niets aan deze afspraak.
- Het dashboard toont een tussenstand volgens deze afspraak. Die tussenstand verandert niets aan de regels of aan het moment van oordeel.
