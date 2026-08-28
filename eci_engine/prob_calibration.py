"""
prob_calibration.py

Pure PROBABILITY-calibratielaag voor Betmobile.

Naamgeving: bewust niet calibration.py. Dat bestand bevat de bestaande
strength-calibratie (build_calibration / apply_calibration) die nu in
productie draait. Deze module kalibreert iets anders: de kansen zelf.
Beide draaien tijdens de schaduwfase naast elkaar; pas bij de bewuste
omschakeling wordt de oude ROI-multiplier-aanpak uitgefaseerd.

Kern:
    p_cal = w * p_model + (1 - w) * p_markt

- De gewichten (w) per competitieklasse staan in een bevroren JSON-bestand,
  gemaakt door fit_calibration.py. Deze module past ze alleen toe.
- Bewust geen database- of config-afhankelijkheden, zodat de module zowel in
  productie (run_model / eci_picks) als in research gebruikt kan worden en
  los te testen is.

Gebruik in de pipeline (schaduwmodus):

    from calibration import load_calibration, calibrate_probs

    calib = load_calibration("output/calibration/calibration_weights.json")
    df = calibrate_probs(df, calib)
    # df heeft nu: prob_cal_home / prob_cal_draw / prob_cal_away,
    # calibration_class, calibration_w, calibration_version

Let op:
- Rijen zonder geldige odds of zonder geldige modelkansen krijgen NaN in de
  prob_cal-kolommen. Behandel dat expliciet (geen pick, of bewuste fallback);
  vul het nooit stilletjes op met de rauwe modelkans.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS_FILENAME = "calibration_weights.json"

# Kolomnamen waaronder ECI-modelkansen kunnen voorkomen, in volgorde van voorkeur.
MODEL_PROB_CANDIDATES = [
    ("prob_home", "prob_draw", "prob_away"),      # productie (picks pipeline)
    ("home_win_pct", "draw_pct", "away_win_pct"),  # historische tuning views
]


# =====================================================================
# GEWICHTENBESTAND
# =====================================================================

def load_calibration(path: str | Path) -> dict:
    """Laad en valideer het bevroren gewichtenbestand."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gewichtenbestand niet gevonden: {path}. "
            "Draai eerst: python fit_calibration.py"
        )

    with open(path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    required = {"version", "classes", "class_patterns", "default_class"}
    missing = required - set(calib)
    if missing:
        raise ValueError(f"Gewichtenbestand mist verplichte velden: {sorted(missing)}")

    return calib


# =====================================================================
# KANSEN VOORBEREIDEN
# =====================================================================

def detect_model_prob_cols(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """Vind de kolommen met ECI-modelkansen (H/D/A)."""
    for cols in MODEL_PROB_CANDIDATES:
        if all(c in df.columns for c in cols):
            return cols
    return None


def normalize_model_probs(
    df: pd.DataFrame,
    prob_cols: tuple[str, str, str],
    out_cols: tuple[str, str, str] = ("mdl_home", "mdl_draw", "mdl_away"),
) -> pd.DataFrame:
    """
    Zet modelkansen om naar een nette 0-1 schaal die per rij op 1 sommeert.

    Accepteert zowel 0-1 als 0-100 invoer; alles daarbuiten wordt NaN.
    """
    df = df.copy()
    probs = df[list(prob_cols)].apply(pd.to_numeric, errors="coerce")
    total = probs.sum(axis=1)

    # Schaal bepalen: som ~1 -> al kansen; som ~100 -> percentages.
    scale = pd.Series(np.nan, index=df.index, dtype="float64")
    scale = scale.mask(total.between(0.90, 1.10), 1.0)
    scale = scale.mask(total.between(90.0, 110.0), 100.0)

    scaled = probs.div(scale, axis=0)
    scaled_total = scaled.sum(axis=1)

    for out_col, col in zip(out_cols, prob_cols):
        df[out_col] = scaled[col] / scaled_total

    return df


def shin_z(imp: np.ndarray, max_iter: int = 60, tol: float = 1e-10) -> np.ndarray:
    """
    Schat Shin's z: het aandeel "geinformeerde inzet" in de markt.

    Shin (1993) modelleert de marge als gevolg van bookmakers die zich
    indekken tegen beter geinformeerde spelers. Dat verklaart waarom de
    marge NIET gelijkmatig over de uitkomsten verdeeld is: op longshots zit
    meer marge dan op favorieten. De proportionele methode negeert dat en
    kent daardoor systematisch te weinig kans toe aan favorieten.

    imp: array (n x k) met 1/odds per uitkomst.
    """
    total = imp.sum(axis=1)
    z = np.zeros(len(imp))
    k = imp.shape[1]

    for _ in range(max_iter):
        disc = z[:, None] ** 2 + 4.0 * (1.0 - z)[:, None] * (imp ** 2) / total[:, None]
        p = (np.sqrt(np.clip(disc, 0, None)) - z[:, None]) / (
            2.0 * np.clip(1.0 - z, 1e-9, None)[:, None]
        )
        s = p.sum(axis=1)
        z_new = np.clip(z + (s - 1.0) / max(k - 1.0, 1e-9), 0.0, 0.9)
        if np.max(np.abs(z_new - z)) < tol:
            z = z_new
            break
        z = z_new
    return z


def pooled_shin_z(odds: np.ndarray, groups: np.ndarray | None = None,
                  min_n: int = 200) -> np.ndarray:
    """
    Schat z per groep (bijv. competitie) in plaats van per wedstrijd.

    Per wedstrijd wordt z uit slechts drie getallen geschat; dat geeft
    ongeveer 12% relatieve ruis. z is echter een eigenschap van de MARKT
    (hoeveel geinformeerd geld er rondgaat), niet van de losse wedstrijd.
    Poolen over een competitie maakt de schatting stabieler.

    Groepen met minder dan min_n wedstrijden vallen terug op het algemene
    gemiddelde.
    """
    imp = 1.0 / odds
    z_row = shin_z(imp)
    out = np.full(len(odds), float(np.nanmedian(z_row)))
    if groups is None:
        return out
    g = pd.Series(groups)
    for name, idx in g.groupby(g, observed=True).groups.items():
        pos = g.index.get_indexer(pd.Index(idx))
        if len(pos) >= min_n:
            out[pos] = float(np.nanmedian(z_row[pos]))
    return out


def probs_from_z(odds: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Bereken Shin-kansen bij een gegeven (bijv. gepoolde) z."""
    imp = 1.0 / odds
    total = imp.sum(axis=1)
    z = np.clip(np.asarray(z, dtype=float), 0.0, 0.9)
    disc = z[:, None] ** 2 + 4.0 * (1.0 - z)[:, None] * (imp ** 2) / total[:, None]
    p = (np.sqrt(np.clip(disc, 0, None)) - z[:, None]) / (
        2.0 * np.clip(1.0 - z, 1e-9, None)[:, None]
    )
    return p / p.sum(axis=1, keepdims=True)


def devig_shin(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ge-devigde kansen volgens Shin. Geeft (kansen, z) terug."""
    imp = 1.0 / odds
    total = imp.sum(axis=1)
    z = shin_z(imp)
    disc = z[:, None] ** 2 + 4.0 * (1.0 - z)[:, None] * (imp ** 2) / total[:, None]
    p = (np.sqrt(np.clip(disc, 0, None)) - z[:, None]) / (
        2.0 * np.clip(1.0 - z, 1e-9, None)[:, None]
    )
    return p / p.sum(axis=1, keepdims=True), z


def compute_market_probs(
    df: pd.DataFrame,
    odds_cols: tuple[str, str, str] = ("odds_home", "odds_draw", "odds_away"),
    out_cols: tuple[str, str, str] = ("mkt_home", "mkt_draw", "mkt_away"),
    method: str = "shin",
    pool_by: str | None = None,
) -> pd.DataFrame:
    """
    Bereken ge-devigde marktkansen uit 1X2 odds.

    method="shin"         : houdt er rekening mee dat de marge zwaarder op
                            longshots drukt. Standaard, omdat de
                            proportionele methode favorieten onderschat -
                            dat gaf in de eerste tier-poging fantoom-EV.
    method="proportional" : imp / som(imp). Bewaard om oude resultaten te
                            kunnen reproduceren.

    Extra kolommen: market_overround en (bij Shin) market_shin_z.
    Rijen met een ongeldige odd (<= 1.01) krijgen NaN.
    """
    df = df.copy()
    odds = df[list(odds_cols)].apply(pd.to_numeric, errors="coerce")
    valid = (odds > 1.01).all(axis=1)

    imp = (1.0 / odds).where(valid)
    total = imp.sum(axis=1).where(valid)

    if method == "proportional":
        for out_col, col in zip(out_cols, odds_cols):
            df[out_col] = imp[col] / total
        df["market_overround"] = total
        return df

    if method != "shin":
        raise ValueError(f"onbekende devig-methode: {method}")

    arr = odds.to_numpy(dtype=float)
    ok = valid.to_numpy()
    probs = np.full(arr.shape, np.nan)
    z_all = np.full(len(arr), np.nan)
    if ok.any():
        if pool_by and pool_by in df.columns:
            # z gepoold per groep: stabieler dan per wedstrijd uit 3 getallen.
            z = pooled_shin_z(arr[ok], df.loc[valid, pool_by].to_numpy())
            p = probs_from_z(arr[ok], z)
        else:
            p, z = devig_shin(arr[ok])
        probs[ok] = p
        z_all[ok] = z

    for i, out_col in enumerate(out_cols):
        df[out_col] = probs[:, i]
    df["market_overround"] = total
    df["market_shin_z"] = z_all
    return df


# =====================================================================
# COMPETITIEKLASSEN
# =====================================================================

def assign_competition_class(competitions: pd.Series, calib: dict) -> pd.Series:
    """
    Bepaal per competitie de calibratieklasse.

    Volgorde:
    1. expliciete mapping uit het gewichtenbestand (competition_class_map),
    2. regex-patronen (class_patterns, eerste match wint),
    3. default_class.
    """
    explicit = {
        str(k).strip().lower(): v
        for k, v in (calib.get("competition_class_map") or {}).items()
    }
    patterns = [
        (name, [re.compile(p, re.IGNORECASE) for p in pats])
        for name, pats in calib["class_patterns"]
    ]
    default = calib["default_class"]

    cache: dict[str, str] = {}

    def resolve(comp: str) -> str:
        key = str(comp).strip().lower()
        if key in cache:
            return cache[key]
        if key in explicit:
            cache[key] = explicit[key]
            return cache[key]
        for name, regs in patterns:
            if any(r.search(key) for r in regs):
                cache[key] = name
                return name
        cache[key] = default
        return default

    return competitions.astype(str).map(resolve)


# =====================================================================
# BLENDEN EN TOEPASSEN
# =====================================================================

def blend_probs(model_mat: np.ndarray, market_mat: np.ndarray, w) -> np.ndarray:
    """
    p_cal = w * model + (1 - w) * markt, daarna per rij hernormaliseren.

    w mag een scalar zijn of een array met een gewicht per rij.
    """
    model_mat = np.asarray(model_mat, dtype=float)
    market_mat = np.asarray(market_mat, dtype=float)

    w_arr = np.asarray(w, dtype=float)
    if w_arr.ndim == 1:
        w_arr = w_arr.reshape(-1, 1)

    blended = w_arr * model_mat + (1.0 - w_arr) * market_mat
    row_sum = blended.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return blended / row_sum


def calibrate_probs(
    df: pd.DataFrame,
    calib: dict,
    competition_col: str = "competition",
    model_prob_cols: tuple[str, str, str] | None = None,
    odds_cols: tuple[str, str, str] = ("odds_home", "odds_draw", "odds_away"),
    out_cols: tuple[str, str, str] = ("prob_cal_home", "prob_cal_draw", "prob_cal_away"),
) -> pd.DataFrame:
    """
    Voeg gekalibreerde kansen toe aan een dataframe met modelkansen en odds.

    Toegevoegde kolommen:
    - prob_cal_home / prob_cal_draw / prob_cal_away (of eigen out_cols)
    - calibration_class : gebruikte competitieklasse
    - calibration_w     : gebruikt gewicht
    - calibration_version : versie van het gewichtenbestand (audit trail)
    """
    df = df.copy()

    if model_prob_cols is None:
        model_prob_cols = detect_model_prob_cols(df)
        if model_prob_cols is None:
            raise ValueError(
                "Geen modelkans-kolommen gevonden. Verwacht een van: "
                f"{MODEL_PROB_CANDIDATES}"
            )

    df = normalize_model_probs(df, model_prob_cols)
    df = compute_market_probs(df, odds_cols)

    df["calibration_class"] = assign_competition_class(df[competition_col], calib)

    w_by_class = {name: float(info["w"]) for name, info in calib["classes"].items()}
    default_w = w_by_class.get(
        calib["default_class"],
        float(calib.get("global", {}).get("w", 0.5)),
    )
    df["calibration_w"] = df["calibration_class"].map(w_by_class).fillna(default_w)

    model_mat = df[["mdl_home", "mdl_draw", "mdl_away"]].to_numpy(dtype=float)
    market_mat = df[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(dtype=float)
    calibrated = blend_probs(model_mat, market_mat, df["calibration_w"].to_numpy())

    for i, col in enumerate(out_cols):
        df[col] = calibrated[:, i]

    df["calibration_version"] = calib["version"]
    return df