"""
tier_assign.py

Kent pick-tiers toe op basis van de BEVROREN EV-definitie uit
tier_rebuild.py, en schrijft er een leesbare uitleg bij.

WAT ER VERANDERT TEN OPZICHTE VAN DE OUDE TIERS
Oud: tier volgde uit segmenten die gevonden waren door honderden
     combinaties af te zoeken op dezelfde data - en optimaliseerde op
     "vaak goed voorspeld", wat samenvalt met lage odds en dus lage winst.
Nieuw: tier volgt uit het historisch gemeten rendement van de prijsklasse
     waarin de weddenschap valt. Uit de validatie: A -2,7% / B -4,0% /
     C -6,1% op de testset, tegen -5,0% voor blind op de favoriet.

DE ROL VAN ECI
ECI bepaalt de tier NIET. Drie onafhankelijke tests lieten zien dat
ECI-kansen niets toevoegen bovenop de markt, en een ECI-opslag maakte de
tier-rangorde in de validatie aantoonbaar slechter. ECI blijft wel in de
uitleg staan als BESCHRIJVING: eens/oneens met de markt, de kans die het
ziet, en het krachtsverschil. "Oneens met de markt" is daarbij een
waarschuwing (historisch -12,4% ROI), niet langer een koopsignaal.

EERLIJKE KANTTEKENING
Alle tiers zijn negatief. De marge (~9%) is groter dan het effect. Dit
rangschikt; het maakt niet winstgevend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Absoluut pad, gebaseerd op de locatie van dit bestand. Met een relatief
# pad vond run_model het bestand niet wanneer hij vanuit een andere map werd
# gestart (bijv. Task Scheduler) en viel hij STIL terug op de oude tiers.
from config import OUTPUT_DIR

DEFAULT_TIER_PATH = OUTPUT_DIR / "calibration" / "tier_definition.json"

_CACHE: dict | None = None
_ECI_CACHE: dict | None = None
_ECI_TRIED = False


def load_eci_recalibration() -> dict | None:
    """Laad de monotone vertaaltabel voor ECI-kansen (indien aanwezig)."""
    global _ECI_CACHE, _ECI_TRIED
    if _ECI_TRIED:
        return _ECI_CACHE
    _ECI_TRIED = True
    try:
        from isotonic import load_calibration_file, DEFAULT_ECI_CALIB_NAME

        _ECI_CACHE = load_calibration_file(
            OUTPUT_DIR / "calibration" / DEFAULT_ECI_CALIB_NAME
        )
    except Exception:  # noqa: BLE001
        _ECI_CACHE = None
    return _ECI_CACHE


def load_tier_definition(path: str | Path | None = None) -> dict | None:
    """Laad de bevroren tier-definitie; geeft None als hij ontbreekt."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    p = Path(path or DEFAULT_TIER_PATH)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        _CACHE = json.load(f)
    return _CACHE


def estimate_ev(odds: float | None, tier_def: dict) -> float | None:
    """Zoek de geschatte EV op basis van de prijsklasse."""
    if odds is None or not np.isfinite(odds) or odds <= 1.01:
        return None
    for c in tier_def.get("ev_curve", []):
        hi = np.inf if c.get("high") is None else c["high"]
        if c["low"] < odds <= hi:
            return float(c["ev"])
    return float(tier_def.get("overall_ev", -0.09))


def ev_to_tier(ev: float | None, tier_def: dict) -> tuple[str, int]:
    """Vertaal EV naar tier en sterren."""
    if ev is None:
        return "C", 2
    stars = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    for name, edge in tier_def.get("tier_edges", []):
        if edge is None or ev >= edge:
            return name, stars.get(name, 2)
    return "D", 1


def build_reason(
    *,
    tier: str,
    ev: float | None,
    odds: float | None,
    market_prob: float | None,
    eci_prob: float | None,
    eci_agrees: bool | None,
    rating_gap: float | None,
    competition: str,
    selection: str,
) -> str:
    """
    Schrijf de uitleg in gewone taal.

    Volgorde: eerst waar de tier vandaan komt (prijsklasse en EV), dan wat
    ECI ervan vindt (beschrijvend), dan het krachtsverschil.
    """
    parts: list[str] = []

    if odds is not None and market_prob is not None:
        parts.append(
            f"{competition}: {selection} tegen {odds:.2f}, markt geeft "
            f"{market_prob:.0%}"
        )
    elif odds is not None:
        parts.append(f"{competition}: {selection} tegen {odds:.2f}")

    if ev is not None:
        parts.append(
            f"prijsklasse leverde historisch {ev:+.1%} op -> tier {tier}"
        )

    if eci_agrees is None or eci_prob is None:
        pass
    elif eci_agrees:
        parts.append(f"ECI is het eens en ziet {eci_prob:.0%}")
    else:
        parts.append(
            f"LET OP: ECI is het oneens (ziet {eci_prob:.0%} voor een andere "
            "uitkomst); die groep deed het historisch juist slechter"
        )

    if rating_gap is not None and np.isfinite(rating_gap):
        if rating_gap >= 1000:
            parts.append(f"groot krachtsverschil (ratinggap {rating_gap:.0f})")
        elif rating_gap >= 500:
            parts.append(f"duidelijk krachtsverschil (ratinggap {rating_gap:.0f})")
        else:
            parts.append(f"klein krachtsverschil (ratinggap {rating_gap:.0f})")

    return " | ".join(parts)


def classify_row(row: pd.Series, tier_def: dict) -> dict:
    """Bepaal tier, EV en uitleg voor een enkele pick."""
    def f(*names):
        for n in names:
            v = row.get(n)
            if v is not None and pd.notna(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    selection = str(row.get("Selection") or row.get("selection") or "").upper()
    odds_map = {
        "HOME": ("odds_home", "Odds Home"),
        "DRAW": ("odds_draw", "Odds Draw"),
        "AWAY": ("odds_away", "Odds Away"),
    }
    prob_map = {
        "HOME": ("home_win_pct", "Home Prob"),
        "DRAW": ("draw_pct", "Draw Prob"),
        "AWAY": ("away_win_pct", "Away Prob"),
    }
    odds = f(*odds_map.get(selection, ()))
    eci_prob = f(*prob_map.get(selection, ()))
    if eci_prob is not None and eci_prob > 1.2:
        eci_prob /= 100.0

    # ECI claimt te extreme kansen (86% waar 81% gebeurt). De monotone
    # vertaaltabel maakt het getoonde getal eerlijk; de rangorde blijft gelijk.
    eci_prob_raw = eci_prob
    eci_cal = load_eci_recalibration()
    if eci_cal and eci_prob is not None:
        from isotonic import apply_isotonic

        eci_prob = float(apply_isotonic(np.array([eci_prob]), eci_cal.get("knots") or [])[0])

    # Marktkans uit alle drie de odds, ge-devigd volgens Shin (dezelfde
    # methode als de rest van het systeem; proportioneel onderschatte de
    # favoriet systematisch).
    o = [f("odds_home", "Odds Home"), f("odds_draw", "Odds Draw"), f("odds_away", "Odds Away")]
    market_prob = None
    if all(x is not None and x > 1.01 for x in o) and odds:
        from prob_calibration import devig_shin

        p_shin, _ = devig_shin(np.array([o], dtype=float))
        idx = {"HOME": 0, "DRAW": 1, "AWAY": 2}.get(selection)
        market_prob = float(p_shin[0, idx]) if idx is not None else None

    # Ziet ECI dezelfde uitkomst als favoriet als de markt?
    eci_all = [
        f("home_win_pct", "Home Prob"), f("draw_pct", "Draw Prob"), f("away_win_pct", "Away Prob")
    ]
    eci_agrees = None
    if all(x is not None for x in eci_all) and all(x is not None and x > 1.01 for x in o):
        labels = ["HOME", "DRAW", "AWAY"]
        eci_fav = labels[int(np.argmax(eci_all))]
        mkt_fav = labels[int(np.argmin(o))]  # laagste odds = favoriet
        eci_agrees = (eci_fav == mkt_fav) and (selection == mkt_fav)

    rating_gap = f("rating_gap")
    if rating_gap is None:
        hr, ar = f("home_rating"), f("away_rating")
        rating_gap = abs(hr - ar) if hr is not None and ar is not None else None

    ev = estimate_ev(odds, tier_def)
    tier, stars = ev_to_tier(ev, tier_def)
    competition = str(row.get("competition") or "onbekend")

    return {
        "eci_prob_raw": eci_prob_raw,
        "eci_prob_calibrated": eci_prob,
        "pick_tier": tier,
        "pick_stars": stars,
        "estimated_ev": ev,
        "tier_version": tier_def.get("version"),
        "classification_reason": build_reason(
            tier=tier, ev=ev, odds=odds, market_prob=market_prob,
            eci_prob=eci_prob, eci_agrees=eci_agrees, rating_gap=rating_gap,
            competition=competition, selection=selection,
        ),
    }


def assign_tiers(picks: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    """
    Voeg de nieuwe tiers toe aan een picks-dataframe.

    Ontbreekt de tier-definitie, dan blijft het frame ongewijzigd - dan
    draait de oude classificatie gewoon door.
    """
    if picks is None or picks.empty:
        return picks
    tier_def = load_tier_definition(path)
    if tier_def is None:
        print(
            f"[tier] WAARSCHUWING: {Path(path or DEFAULT_TIER_PATH)} niet gevonden.\n"
            "[tier] De OUDE tiers blijven staan. Draai tier_rebuild.py om de\n"
            "[tier] definitie aan te maken, of controleer het pad."
        )
        return picks

    out = picks.copy()
    res = [classify_row(r, tier_def) for _, r in out.iterrows()]
    for key in ("pick_tier", "pick_stars", "estimated_ev", "tier_version",
                "classification_reason", "eci_prob_raw", "eci_prob_calibrated"):
        out[key] = [r[key] for r in res]
    return out

# Volgorde van best naar slechtst. Bevat OOK de oude namen (A-, X), zodat
# picks uit een eerdere versie nog netjes gesorteerd worden.
TIER_ORDER = {"A+": 0, "A": 1, "A-": 2, "B": 3, "C": 4, "D": 5, "X": 6}


def sort_by_tier(picks: pd.DataFrame) -> pd.DataFrame:
    """
    Sorteer picks op de HUIDIGE tier, daarna op geschatte EV.

    Moet ná assign_tiers draaien: classify_picks sorteert op de oude tier en
    die kolom wordt daarna overschreven.
    """
    if picks is None or picks.empty or "pick_tier" not in picks.columns:
        return picks
    out = picks.copy()
    out["_tier_order"] = out["pick_tier"].map(TIER_ORDER).fillna(9)
    out["_ev"] = pd.to_numeric(out.get("estimated_ev"), errors="coerce").fillna(-1.0)
    out = out.sort_values(["_tier_order", "_ev"], ascending=[True, False])
    return out.drop(columns=["_tier_order", "_ev"])