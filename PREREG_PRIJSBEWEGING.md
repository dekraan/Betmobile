# Pre-registratie H-BEW: voorspelt de ECI-selectie een gunstiger prijsbeweging?

Vastgelegd op 23-10-2026, vóór enige analyse van prijsbewegingen in de meetperiode (wedstrijden vanaf 18-08-2026). Dit document staat los van `MEETAFSPRAAK.md` en verandert daar niets aan.

## Aanleiding (hypothesegenererend, telt niet als bewijs)

De historische analyse over 1 juli t/m 17 augustus 2026 (`research_selectie_vergelijking.py`, deel 3) gaf voor Beide − Alleen B:

| Maat | Verschil | 95%-interval |
| --- | --- | --- |
| CLV | +2,66 pp | [−0,69; +6,02] |
| marge bij instap | +0,88 pp | [+0,47; +1,28], t = 4,37 |
| beweging | +1,79 pp | [−1,53; +5,10] |
| beweging, gecorrigeerd voor oddsband | +1,71 pp | [−2,05; +5,47] |

Het deel van het CLV-verschil dat statistisch hard is, zit in de marge bij instap, dus in het prijsniveau. Voor de beweging blijft een positief, maar onzeker verschil over.

Daarnaast werd Beide in die analyse ingedeeld op basis van de keuze van de ECI-regel in een willekeurige, mogelijk latere run. Het lidmaatschap kan dan afhangen van wat er na het instapmoment gebeurde. Deze toets gebruikt daarom een striktere indeling: zie "Kandidaten".

## Hypothese

**H-BEW:** kandidaten die de bevroren ECI-regel selecteert op het moment van instap, krijgen tot de slotkoers een gunstiger prijsbeweging dan vergelijkbare kandidaten die hij op dat moment niet selecteert.

## Kandidaten

- **Bron:** `model_match_snapshots`, wedstrijddatum vanaf 18-08-2026 tot en met de wedstrijddatum van de 150e afgerekende MAIN-pick volgens `MEETAFSPRAAK.md`.
- **Kandidaat:** elke combinatie van wedstrijd en kant (thuis of uit) op de eerste modelrun waarop het marktdeel van de bevroren regels geldt:
  - odds van die kant tussen 1,50 en 2,50
  - minstens 7 snapshots
  - drift ≤ +3%
- **ECI-selectie:** de bevroren MAIN-regel kiest die kant op diezelfde run. Dat is de regel met kans ≥ 0,58, value ≥ 1,02, rating gap ≥ 500, thuis-edge en strength ≥ 1,8, met calibratie uit.
- **Vergelijkingsgroep:** alle overige kandidaten. Dat geldt ook als de ECI-regel ze later alsnog kiest, want op het instapmoment waren ze niet geselecteerd.
- **Uitsluitingen,** voor beide groepen gelijk:
  - geen 3-weg odds op de instaprun
  - geen Bet365-snapshot op of vóór de aftrap
  - wedstrijd niet gespeeld (status anders dan FT, AET of PEN)
- **Geldigheid:** de regelvlaggen en de MAIN-picks moeten elk voor minstens 99,5% te reconstrueren zijn uit de opgeslagen waarden. Lukt dat niet, dan wordt de toets niet uitgevoerd.

## Uitkomstmaat

**beweging = (p_close − p_instap) × odds_instap**

- `p_instap`: de Shin-kans van de kant, uit de Bet365-odds die het model op de instaprun zag.
- `p_close`: de Shin-kans van de kant, uit de laatste Bet365-snapshot op of vóór de aftrap (`odds_1x2_bet365_snapshots_mv`).
- Een positieve waarde betekent dat de markt na instap richting deze kant bewoog.

## Toets

- **Model:** OLS van de beweging op een constante, een dummy voor ECI-selectie, en dummy's voor de oddsband van de instapodds (1,50–1,65 / 1,65–1,80 / 1,80–2,00 / 2,00–2,50).
- **Primaire schatter:** de coëfficiënt op de ECI-dummy, dus het verschil binnen dezelfde oddsband.
- **Standaardfout:** cluster-robuust per competitie (CR1), met kritieke waarde t(G−1), tweezijdig, α = 0,05.
- Er is één primaire toets, dus een correctie voor meervoudig toetsen is niet nodig.

## Relevantie

Een verschil telt als relevant vanaf **+1,0 procentpunt** beweging. Een kleiner verschil is van dezelfde orde als het mechanische margeverschil uit de historische analyse (+0,9 pp). Het verandert in de praktijk weinig aan een selectie die op ongeveer −4% CLV zit.

## Beslisregel

| Uitkomst | Voorwaarde |
| --- | --- |
| Bevestigd | ondergrens > 0 én schatting ≥ +1,0 pp |
| Aangetoond, maar klein | ondergrens > 0 én schatting < +1,0 pp |
| Geen relevant effect | bovengrens < +1,0 pp |
| Onbeslist | in alle andere gevallen |

## Steekproef en moment

- Er is **één analyse**, op het moment dat de 150e MAIN-pick is afgerekend. Er worden geen tussentijdse analyses van de beweging gedaan.
- De verwachte omvang is naar schatting 100 tot 140 ECI-kandidaten op instap. Het controleblok in `research_selectie_vergelijking.py` laat zien welk deel van de historische picks al op het instapmoment werd gekozen.
- **Verwachte gevoeligheid,** geschat op basis van de historische standaardfout (1,86 pp bij n = 32):
  - Is het echte effect zo groot als de historische schatting (+1,7 pp), dan is de kans op "Bevestigd" ongeveer 35–45%.
  - Bij een echt effect van +1,0 pp is die kans ongeveer 15–20%.
  - Is er geen effect, dan is "Geen relevant effect" met deze omvang ook maar in ongeveer 15–20% van de gevallen bereikbaar.

  "Onbeslist" is dus een reële uitkomst. De historische schatting is bovendien waarschijnlijk te gunstig, omdat juist die uitkomst tot deze toets leidde.
- Wie een grotere steekproef wil, moet dat nu vastleggen, niet na een eerste blik op de data.

## Wat niet mag

- Na 23-10-2026 niets meer veranderen aan de definities, de oddsbanden, de toets, de relevantiegrens of de steekproef.
- De historische en de nieuwe data niet samenvoegen. De historische uitkomst telt niet mee.
- De uitkomst niet gebruiken om regels te ontwikkelen en diezelfde regels daarna als bevestigd te presenteren.
- Bij "Onbeslist" niet doorgaan tot het wel uitkomt. Een vervolg vraagt een nieuwe pre-registratie op nieuwe data.

## Rapportage

Deze toets wordt apart gerapporteerd van het oordeel over de 150 MAIN-picks uit `MEETAFSPRAAK.md`. Die evaluatie beoordeelt de prestaties van de selectie zelf, op CLV en ROI. Deze toets beantwoordt alleen de vraag of de ECI-selectie een gunstiger prijsbeweging voorspelt.

## Code

`toets_prijsbeweging.py`, SHA-256 met regeleinden genormaliseerd naar LF:

`f5c00cc989b67f73603d5078d38ccd7016997fb7ac320d5f4c72fa5841e52f41`

Het script print bij elke run zijn eigen hash. Alleen een run met deze hash telt als de vastgelegde toets. Tot het eindpunt is bereikt, weigert het script te draaien.
