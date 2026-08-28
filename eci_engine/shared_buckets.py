"""
shared_buckets.py

Eén plek voor de bucket-indelingen die in meerdere rapporten voorkomen.

Waarom: tier_rebuild gebruikte tien odds-buckets, clv_report en
betmobile_report zes, en research_backtest weer andere grenzen. Tabellen uit
verschillende scripts waren daardoor niet naast elkaar te leggen.

De fijne indeling (ODDS_BINS_FINE) is die van de EV-curve; de grove
(ODDS_BINS_REPORT) is voor rapportage over kleinere aantallen picks.
"""

from __future__ import annotations

import numpy as np

# Fijn: voor het meten van rendement per prijsklasse (veel waarnemingen).
ODDS_BINS_FINE = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 6.0, np.inf]
ODDS_LABELS_FINE = ["<1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0",
                    "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0-6.0", "6.0+"]

# Grof: voor pickrapportage, waar de aantallen per bucket klein zijn.
ODDS_BINS_REPORT = [1.0, 1.6, 1.8, 2.0, 2.2, 2.5, np.inf]
ODDS_LABELS_REPORT = ["1.0-1.6", "1.6-1.8", "1.8-2.0", "2.0-2.2", "2.2-2.5", "2.5+"]

# Kansbuckets (ECI en markt), gebruikt in kalibratietabellen.
PROB_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]

# Ratingverschil.
GAP_BINS = [0, 100, 250, 500, 1000, 100000]
GAP_LABELS = ["0-100", "100-250", "250-500", "500-1000", "1000+"]