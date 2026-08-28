"""
isotonic.py

Isotone regressie (Pool Adjacent Violators) en twee toepassingen:

1. ECI-HERCALIBRATIE
   ECI rangschikt correct (monotonie 100%) maar de niveaus kloppen niet:
   waar ECI 86% claimt gebeurt het 81% van de tijd. Isotone regressie leert
   een monotone vertaaltabel "wat ECI zegt -> wat er werkelijk gebeurt",
   zonder de rangorde aan te tasten. Het getal op het scherm wordt eerlijk;
   je krijgt er geen edge bij.

2. GLADDE EV-CURVE
   De EV per prijsklasse werd gemeten met tien handmatige odds-buckets. Dat
   gaf sprongen tot 5,9 procentpunt bij een bucketrand: een wedstrijd op
   odds 1.79 kreeg een wezenlijk andere EV dan één op 1.81. Isotone
   regressie met dalende beperking legt de breekpunten waar de DATA ze legt,
   niet waar iemand een grens trok. De dalende richting volgt uit de
   favourite-longshot bias: hogere odds horen een lager rendement te geven.

Geen externe afhankelijkheden: PAVA is hier zelf geimplementeerd, zodat er
geen scikit-learn nodig is.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# KERN: POOL ADJACENT VIOLATORS
# =====================================================================

def pava(y: np.ndarray, w: np.ndarray | None = None, increasing: bool = True) -> np.ndarray:
    """
    Isotone regressie via Pool Adjacent Violators.

    Geeft de best passende monotone reeks bij y (gewogen met w). Waar de
    reeks de monotonie schendt, worden aangrenzende punten samengevoegd tot
    hun gewogen gemiddelde - net zolang tot alles netjes loopt.

    y moet al gesorteerd zijn op de verklarende variabele.
    """
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if not increasing:
        y = -y

    # Blokken: (som van w*y, som van w)
    vals: list[float] = []
    wts: list[float] = []
    for yi, wi in zip(y, w):
        vals.append(yi * wi)
        wts.append(wi)
        # Voeg samen zolang het vorige blok hoger ligt dan het huidige
        while len(vals) > 1 and vals[-2] / wts[-2] > vals[-1] / wts[-1]:
            v = vals.pop() + vals[-1]
            ww = wts.pop() + wts[-1]
            vals[-1], wts[-1] = v, ww

    out = np.empty(len(y))
    i = 0
    for v, ww in zip(vals, wts):
        # Hoeveel oorspronkelijke punten zaten in dit blok?
        n_block = 0
        acc = 0.0
        j = i
        while j < len(y) and abs(acc - ww) > 1e-12:
            acc += w[j]
            n_block += 1
            j += 1
        out[i:i + n_block] = v / ww
        i += n_block

    return -out if not increasing else out


def fit_isotonic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    increasing: bool = True,
    n_points: int = 60,
) -> list[tuple[float, float]]:
    """
    Fit een monotoon verband y(x) en geef het terug als knikpunten.

    De punten worden eerst gebundeld in kwantielgroepen (n_points), zodat de
    curve compact op te slaan is en niet elke waarneming een eigen knik krijgt.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 20:
        return []

    order = np.argsort(x)
    x, y, w = x[order], y[order], w[order]

    # Bundel in groepen van gelijke omvang
    n_points = max(5, min(n_points, len(x) // 20))
    edges = np.quantile(x, np.linspace(0, 1, n_points + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)

    gx, gy, gw = [], [], []
    for g in range(len(edges) - 1):
        m = idx == g
        if not m.any():
            continue
        gw.append(w[m].sum())
        gx.append(np.average(x[m], weights=w[m]))
        gy.append(np.average(y[m], weights=w[m]))

    fitted = pava(np.array(gy), np.array(gw), increasing=increasing)
    return [(float(a), float(b)) for a, b in zip(gx, fitted)]


def apply_isotonic(x: np.ndarray, knots: list[tuple[float, float]]) -> np.ndarray:
    """Pas een gefitte curve toe met lineaire interpolatie tussen de knikken."""
    if not knots:
        return np.asarray(x, dtype=float)
    kx = np.array([k[0] for k in knots], dtype=float)
    ky = np.array([k[1] for k in knots], dtype=float)
    return np.interp(np.asarray(x, dtype=float), kx, ky)


# =====================================================================
# TOEPASSING 1: ECI-HERCALIBRATIE
# =====================================================================

DEFAULT_ECI_CALIB_NAME = "eci_isotonic.json"


def fit_eci_recalibration(df: pd.DataFrame) -> dict:
    """
    Leer de vertaaltabel van ECI-kans naar werkelijke kans.

    Elke wedstrijd levert drie waarnemingen (thuis/gelijk/uit): de door ECI
    geclaimde kans en of het ook echt gebeurde.
    """
    xs, ys = [], []
    for i, side in enumerate(["home", "draw", "away"]):
        xs.append(df[f"mdl_{side}"].to_numpy(float))
        ys.append((df["y_idx"].to_numpy(int) == i).astype(float))
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    knots = fit_isotonic(x, y, increasing=True)
    return {
        "version": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "n_observations": int(len(x)),
        "knots": knots,
    }


def recalibrate_eci(probs: np.ndarray, calib: dict) -> np.ndarray:
    """
    Vertaal ECI-kansen naar gekalibreerde kansen en normaliseer per wedstrijd.

    probs: array (n x 3). De rangorde binnen een wedstrijd blijft behouden
    omdat de vertaling monotoon is.
    """
    knots = calib.get("knots") or []
    if not knots:
        return probs
    out = apply_isotonic(np.asarray(probs, dtype=float).ravel(), knots)
    out = out.reshape(np.asarray(probs).shape)
    out = np.clip(out, 1e-4, 1 - 1e-4)
    return out / out.sum(axis=1, keepdims=True)


def save_calibration(calib: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)


def load_calibration_file(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)