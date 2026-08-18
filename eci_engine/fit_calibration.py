"""
fit_calibration.py

Fit de shrinkage-gewichten voor de Betmobile calibratielaag.

    p_cal = w * p_model + (1 - w) * p_markt

Belangrijk:
- Gefit op de VOLLEDIGE wedstrijdenset (historische bronview), niet op picks.
  Calibratie meten op picks heeft ingebouwde selectiebias.
- De gewichten worden per competitieklasse bepaald via grid search op log loss,
  met een train/test-splitsing op datum als stabiliteitscheck.
- Het resultaat is een bevroren, geversioneerd JSON-bestand dat door
  calibration.py wordt toegepast. Herfitten gebeurt bewust en gepland
  (seizoensgrenzen), niet automatisch.

Gebruik:
    python fit_calibration.py                  # fit + gewichtenbestand + rapporten
    python fit_calibration.py --export-csv     # idem, met CSV-exports
    python fit_calibration.py --check          # monitoring van bevroren gewichten
    python fit_calibration.py --check --since 2026-08-01
    python fit_calibration.py --source andere_view --no-refresh

Na de eerste fit ALTIJD controleren:
1. De COMPETITIE -> KLASSE tabel: kloppen de toewijzingen? Pas zo nodig
   CLASS_PATTERNS hieronder aan en fit opnieuw.
2. De W-waarden per klasse: lage w = weinig vertrouwen in ECI daar.
3. De reliability-tabellen: na calibratie hoort avg_prob dicht bij hitrate.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import OUTPUT_DIR
from db import db_engine, refresh_source_views

from prob_calibration import (
    DEFAULT_WEIGHTS_FILENAME,
    assign_competition_class,
    blend_probs,
    compute_market_probs,
    detect_model_prob_cols,
    normalize_model_probs,
)


# =====================================================================
# CONFIG
# =====================================================================

DEFAULT_SOURCE = "betmobile_tuning_preko_mv"
DEFAULT_SCHEMA = "public"

CALIB_DIR = OUTPUT_DIR / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_WEIGHTS_PATH = CALIB_DIR / DEFAULT_WEIGHTS_FILENAME

# Grid search: w van 0.00 (puur markt) t/m 1.00 (puur ECI) in stappen van 0.05.
W_GRID = [round(w, 2) for w in np.arange(0.0, 1.0001, 0.05)]

# Datumsplitsing: eerste deel fit, laatste deel test.
TRAIN_FRAC = 0.65

# Klassen met minder train-wedstrijden vallen terug op het globale gewicht.
MIN_CLASS_MATCHES = 400

# --check: minimaal aantal nieuwe wedstrijden per klasse voor een oordeel.
MIN_CHECK_MATCHES = 150

# --check drempels:
# ATTENTIE als calibratie slechter is dan puur markt (met kleine marge), of
# als de log loss duidelijk verslechterd is t.o.v. de referentie uit de fit.
CHECK_MAX_DELTA_VS_MARKET = 0.005
CHECK_MAX_DRIFT_VS_REF = 0.020

# ---------------------------------------------------------------------
# Competitieklassen.
#
# Eerste match wint (van boven naar beneden), rest valt in DEFAULT_CLASS.
# LET OP na de eerste run: controleer de mappingtabel in de output.
# Bekende valkuil: r"serie a" matcht ook "Brazil Serie A" (kalenderjaar-
# competitie). Verplaats zulke gevallen door het patroon aan te scherpen of
# door ze expliciet in een andere klasse te patternen.
# ---------------------------------------------------------------------
DEFAULT_CLASS = "standard"

CLASS_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "summer_league",
        [
            r"norway|norwegian|eliteserien|obos",
            r"sweden|swedish|allsvenskan|superettan",
            r"finland|finnish|veikkausliiga|ykk",
            r"iceland|icelandic|besta deild|urvalsdeild",
            r"\bireland\b|irish",
            r"faroe",
            r"estonia|meistriliiga",
            r"latvia|virsl",
            r"lithuania|a lyga",
        ],
    ),
    (
        "top",
        [
            r"premier league",
            r"la ?liga",
            r"\bserie a\b",
            r"\bbundesliga\b",
            r"ligue 1",
            r"eredivisie",
            r"primeira liga",
            r"champions league",
            r"europa league",
        ],
    ),
]


# =====================================================================
# PRINT HELPERS
# =====================================================================

def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def print_table(title: str, df: pd.DataFrame, max_rows: int = 60) -> None:
    print_header(title)
    if df is None or df.empty:
        print("Geen data.")
        return
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].round(4)
    print(view.to_string(index=False))


# =====================================================================
# DATA LADEN
# =====================================================================

def get_table_columns(source: str, schema: str = DEFAULT_SCHEMA) -> set[str]:
    """Lees beschikbare kolommen uit de PostgreSQL-catalogus."""
    q = text(
        """
        SELECT a.attname AS column_name
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema
          AND c.relname = :source
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """
    )
    with db_engine().connect() as conn:
        rows = conn.execute(q, {"schema": schema, "source": source}).fetchall()
    cols = {r[0] for r in rows}
    if not cols:
        raise RuntimeError(
            f"Bron {schema}.{source} niet gevonden of zonder kolommen. "
            "Controleer de naam."
        )
    return cols


# Kolommen die we willen meenemen als ze bestaan.
WANTED_COLUMNS = [
    "match_id",
    "competition",
    "date",
    "home_team",
    "away_team",
    "odds_home",
    "odds_draw",
    "odds_away",
    "home_win_pct",
    "draw_pct",
    "away_win_pct",
    "prob_home",
    "prob_draw",
    "prob_away",
    "result",
    "result_code",
    "final_result",
    "score",
    "home_score",
    "away_score",
]

REQUIRED_COLUMNS = {"competition", "date", "odds_home", "odds_draw", "odds_away"}


def load_match_frame(source: str, schema: str = DEFAULT_SCHEMA) -> pd.DataFrame:
    """Laad de volledige wedstrijdenset uit de bronview."""
    available = get_table_columns(source, schema)

    missing_required = REQUIRED_COLUMNS - available
    if missing_required:
        raise RuntimeError(
            f"Bron {schema}.{source} mist verplichte kolommen: {sorted(missing_required)}"
        )

    select_cols = [c for c in WANTED_COLUMNS if c in available]
    q = f"SELECT {', '.join(select_cols)} FROM {schema}.{source}"

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    print(f"[load] {schema}.{source}: {len(df)} rijen, kolommen: {select_cols}")
    return df


# =====================================================================
# UITSLAGEN NORMALISEREN
# =====================================================================

RESULT_MAP = {
    "H": "H", "D": "D", "A": "A",
    "HOME": "H", "DRAW": "D", "AWAY": "A",
    "HOME_WIN": "H", "HOME WIN": "H",
    "AWAY_WIN": "A", "AWAY WIN": "A",
    "DRAWN": "D",
    "1": "H", "X": "D", "2": "A",
}
SCORE_RE = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$")


def _result_from_scores(home, away, index) -> pd.Series:
    hs = pd.to_numeric(home, errors="coerce")
    aw = pd.to_numeric(away, errors="coerce")
    arr = np.select([hs > aw, hs == aw, hs < aw], ["H", "D", "A"], default=None)
    return pd.Series(arr, index=index, dtype="object")


def normalize_results(df: pd.DataFrame) -> pd.Series:
    """Zet uitslagen om naar H/D/A, uit codekolommen of scores."""
    res = pd.Series(np.nan, index=df.index, dtype="object")

    for col in ("result_code", "result", "final_result"):
        if col not in df.columns:
            continue
        raw = df[col].astype(str).str.strip().str.upper()
        mapped = raw.map(RESULT_MAP)

        # Scorestrings zoals "2-1" of "2 : 1".
        need = mapped.isna()
        if need.any():
            parts = raw[need].str.extract(SCORE_RE)
            mapped = mapped.fillna(
                _result_from_scores(parts[0], parts[1], parts.index)
            )
        res = res.fillna(mapped)

    if {"home_score", "away_score"} <= set(df.columns):
        res = res.fillna(
            _result_from_scores(df["home_score"], df["away_score"], df.index)
        )

    # Scorekolom zoals "2-1" of "2 : 1" (zelfde fallback als research_backtest).
    if "score" in df.columns:
        parts = df["score"].astype(str).str.strip().str.extract(SCORE_RE)
        res = res.fillna(_result_from_scores(parts[0], parts[1], df.index))

    return res


# =====================================================================
# FRAME VOORBEREIDEN
# =====================================================================

def _patterns_calib_stub(
    class_patterns: list[tuple[str, list[str]]],
    default_class: str,
    competition_class_map: dict | None = None,
) -> dict:
    """Minimale calib-dict zodat we assign_competition_class kunnen hergebruiken."""
    return {
        "class_patterns": [[name, list(pats)] for name, pats in class_patterns],
        "default_class": default_class,
        "competition_class_map": competition_class_map or {},
    }


def prepare_match_frame(
    raw: pd.DataFrame,
    class_calib: dict | None = None,
) -> pd.DataFrame:
    """
    Maak de wedstrijdenset klaar voor fitten:
    datum, uitslagcode, model- en marktkansen, competitieklasse.
    """
    df = raw.copy()
    n_start = len(df)

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["result_code_norm"] = normalize_results(df)

    prob_cols = detect_model_prob_cols(df)
    if prob_cols is None:
        raise RuntimeError(
            "Geen modelkans-kolommen gevonden in de bron "
            "(verwacht home_win_pct/draw_pct/away_win_pct of prob_home/prob_draw/prob_away)."
        )

    df = normalize_model_probs(df, prob_cols)
    df = compute_market_probs(df)

    mask_date = df["date_dt"].notna()
    mask_result = df["result_code_norm"].isin(["H", "D", "A"])
    mask_model = df[["mdl_home", "mdl_draw", "mdl_away"]].notna().all(axis=1)
    mask_market = df[["mkt_home", "mkt_draw", "mkt_away"]].notna().all(axis=1)

    print(
        f"[prepare] start={n_start} | geen datum={int((~mask_date).sum())} | "
        f"geen uitslag={int((~mask_result).sum())} | "
        f"ongeldige modelkansen={int((~mask_model).sum())} | "
        f"ongeldige odds={int((~mask_market).sum())}"
    )

    df = df[mask_date & mask_result & mask_model & mask_market].copy()
    df["y_idx"] = df["result_code_norm"].map({"H": 0, "D": 1, "A": 2}).astype(int)

    if class_calib is None:
        class_calib = _patterns_calib_stub(CLASS_PATTERNS, DEFAULT_CLASS)
    df["calibration_class"] = assign_competition_class(df["competition"], class_calib)

    print(f"[prepare] bruikbare wedstrijden: {len(df)}")
    print(
        "[prepare] verdeling per klasse: "
        f"{df['calibration_class'].value_counts().to_dict()}"
    )
    return df


# =====================================================================
# METRIEKEN
# =====================================================================

def _matrices(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_mat = part[["mdl_home", "mdl_draw", "mdl_away"]].to_numpy(dtype=float)
    market_mat = part[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(dtype=float)
    y = part["y_idx"].to_numpy(dtype=int)
    return model_mat, market_mat, y


def multiclass_logloss(prob_mat: np.ndarray, y: np.ndarray) -> float:
    p = prob_mat[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(p, 1e-9, None))))


def multiclass_brier(prob_mat: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(prob_mat)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((prob_mat - onehot) ** 2, axis=1)))


def logloss_at_w(model_mat, market_mat, y, w: float) -> float:
    return multiclass_logloss(blend_probs(model_mat, market_mat, w), y)


def logloss_curve(part: pd.DataFrame, w_grid: list[float]) -> pd.DataFrame:
    """Log loss voor elk gewicht in het grid."""
    model_mat, market_mat, y = _matrices(part)
    rows = [
        {"w": w, "logloss": logloss_at_w(model_mat, market_mat, y, w)}
        for w in w_grid
    ]
    return pd.DataFrame(rows)


def best_w(curve: pd.DataFrame) -> float:
    """Laagste log loss wint; bij gelijkspel wint de laagste w (conservatiever)."""
    return float(curve.sort_values(["logloss", "w"]).iloc[0]["w"])


# =====================================================================
# FITTEN
# =====================================================================

def fit_one(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    w_grid: list[float],
    forced_w: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Fit (of forceer) een gewicht en evalueer op de testset.

    forced_w wordt gebruikt voor de fallback naar het globale gewicht.
    """
    curve_train = logloss_curve(train, w_grid)
    w = forced_w if forced_w is not None else best_w(curve_train)

    info: dict = {
        "w": w,
        "n_matches": int(len(train) + len(test)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "fallback_to_global": forced_w is not None,
    }

    if len(test):
        model_mat, market_mat, y = _matrices(test)
        cal_mat = blend_probs(model_mat, market_mat, w)
        curve_test = logloss_curve(test, w_grid)
        info["w_best_test"] = best_w(curve_test)
        info["logloss_test"] = {
            "model": multiclass_logloss(model_mat, y),
            "market": multiclass_logloss(market_mat, y),
            "calibrated": multiclass_logloss(cal_mat, y),
        }
        info["brier_test"] = {
            "model": multiclass_brier(model_mat, y),
            "market": multiclass_brier(market_mat, y),
            "calibrated": multiclass_brier(cal_mat, y),
        }
    else:
        curve_test = pd.DataFrame(columns=["w", "logloss"])
        info["w_best_test"] = None
        info["logloss_test"] = None
        info["brier_test"] = None

    curve_train = curve_train.rename(columns={"logloss": "logloss_train"})
    curve_test = curve_test.rename(columns={"logloss": "logloss_test"})
    curve = curve_train.merge(curve_test, on="w", how="left")
    curve.insert(0, "class", name)

    return info, curve


def fit_all(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    w_grid: list[float] | None = None,
    min_class_matches: int = MIN_CLASS_MATCHES,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Fit het globale gewicht en de gewichten per klasse.

    Splitsing gebeurt op EEN gezamenlijke datum (quantile van alle wedstrijden),
    zodat train altijd strikt voor test ligt, ook per klasse.
    """
    w_grid = w_grid or W_GRID

    split_date = df["date_dt"].quantile(train_frac)
    is_train = df["date_dt"] <= split_date
    print(
        f"[fit] splitsdatum={pd.Timestamp(split_date).date()} | "
        f"train={int(is_train.sum())} | test={int((~is_train).sum())}"
    )

    global_info, global_curve = fit_one(
        "GLOBAL", df[is_train], df[~is_train], w_grid
    )

    class_infos: dict[str, dict] = {}
    curves = [global_curve]

    for name, part in df.groupby("calibration_class", observed=True):
        train = part[part["date_dt"] <= split_date]
        test = part[part["date_dt"] > split_date]

        forced = global_info["w"] if len(train) < min_class_matches else None
        info, curve = fit_one(str(name), train, test, w_grid, forced_w=forced)
        class_infos[str(name)] = info
        curves.append(curve)

    summary_rows = []
    for name, info in [("GLOBAL", global_info)] + sorted(class_infos.items()):
        row = {
            "class": name,
            "n_matches": info["n_matches"],
            "n_train": info["n_train"],
            "n_test": info["n_test"],
            "w": info["w"],
            "fallback": info["fallback_to_global"],
            "w_best_test": info["w_best_test"],
        }
        if info["logloss_test"]:
            ll = info["logloss_test"]
            row["ll_test_model"] = ll["model"]
            row["ll_test_market"] = ll["market"]
            row["ll_test_cal"] = ll["calibrated"]
            row["cal_vs_market"] = ll["calibrated"] - ll["market"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    wgrid_table = pd.concat(curves, ignore_index=True)

    weights = {"global": global_info, "classes": class_infos}
    return weights, summary, wgrid_table, pd.Timestamp(split_date)


# =====================================================================
# RELIABILITY (SANITY CHECK)
# =====================================================================

RELIABILITY_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]


def build_reliability(
    test_df: pd.DataFrame,
    w_by_class: dict[str, float],
    default_w: float,
) -> pd.DataFrame:
    """
    Reliability-tabel op de testset: voorspelde kans vs werkelijke frequentie.

    Elke wedstrijd levert drie observaties (H, D, A). Na goede calibratie hoort
    'gap' (hitrate - avg_prob) per bucket dicht bij nul te liggen.
    """
    if test_df.empty:
        return pd.DataFrame()

    model_mat, market_mat, y = _matrices(test_df)
    w_arr = (
        test_df["calibration_class"].map(w_by_class).fillna(default_w).to_numpy()
    )
    cal_mat = blend_probs(model_mat, market_mat, w_arr)

    onehot = np.zeros_like(model_mat)
    onehot[np.arange(len(y)), y] = 1.0

    frames = []
    for source, mat in [("model", model_mat), ("market", market_mat), ("calibrated", cal_mat)]:
        frames.append(
            pd.DataFrame(
                {
                    "class": np.repeat(test_df["calibration_class"].to_numpy(), 3),
                    "source": source,
                    "prob": mat.ravel(),
                    "hit": onehot.ravel(),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)

    long["bucket"] = pd.cut(long["prob"], bins=RELIABILITY_BINS, include_lowest=True)
    out = (
        long.groupby(["class", "source", "bucket"], observed=True)
        .agg(n=("hit", "size"), avg_prob=("prob", "mean"), hitrate=("hit", "mean"))
        .reset_index()
    )
    out["gap"] = out["hitrate"] - out["avg_prob"]
    out["bucket"] = out["bucket"].astype(str)
    return out


# =====================================================================
# FIT RUN
# =====================================================================

def run_fit(
    source: str = DEFAULT_SOURCE,
    schema: str = DEFAULT_SCHEMA,
    train_frac: float = TRAIN_FRAC,
    min_class_matches: int = MIN_CLASS_MATCHES,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    export_csv: bool = False,
    refresh: bool = True,
    df: pd.DataFrame | None = None,
) -> dict:
    print_header("BETMOBILE CALIBRATION FIT")
    print(
        f"source={source} | train_frac={train_frac} | "
        f"min_class_matches={min_class_matches} | w_grid_step=0.05"
    )

    if df is None:
        if refresh:
            try:
                refresh_source_views()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] refresh_source_views mislukt, ga door: {exc}")
        raw = load_match_frame(source, schema)
        df = prepare_match_frame(raw)
    else:
        df = prepare_match_frame(df)

    if df.empty:
        raise RuntimeError("Geen bruikbare wedstrijden om op te fitten.")

    # Mappingtabel: dit MOET je na de eerste run controleren.
    mapping = (
        df.groupby(["calibration_class", "competition"], observed=True)
        .size()
        .reset_index(name="matches")
        .sort_values(["calibration_class", "matches"], ascending=[True, False])
    )

    weights, summary, wgrid_table, split_date = fit_all(
        df,
        train_frac=train_frac,
        min_class_matches=min_class_matches,
    )

    w_by_class = {name: info["w"] for name, info in weights["classes"].items()}
    default_w = w_by_class.get(DEFAULT_CLASS, weights["global"]["w"])
    reliability = build_reliability(
        df[df["date_dt"] > split_date], w_by_class, default_w
    )

    # ---------------- Gewichtenbestand ----------------
    version = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    calib = {
        "version": version,
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "source": f"{schema}.{source}",
        "data_window": {
            "start": str(df["date_dt"].min().date()),
            "end": str(df["date_dt"].max().date()),
        },
        "split_date": str(split_date.date()),
        "train_frac": train_frac,
        "min_class_matches": min_class_matches,
        "default_class": DEFAULT_CLASS,
        "class_patterns": [[name, list(pats)] for name, pats in CLASS_PATTERNS],
        "competition_class_map": {
            str(comp): str(cls)
            for cls, comp in mapping[["calibration_class", "competition"]].itertuples(
                index=False
            )
        },
        "global": weights["global"],
        "classes": weights["classes"],
    }

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    # ---------------- Rapportage ----------------
    print_table("COMPETITIE -> KLASSE (controleer dit!)", mapping, max_rows=120)
    print_table("FIT SUMMARY (log loss: lager = beter)", summary)

    print_header("GEWICHTEN IN GEWONE TAAL")
    for name, info in sorted(weights["classes"].items()):
        extra = " (fallback naar globaal gewicht: te weinig data)" if info["fallback_to_global"] else ""
        print(
            f"- {name}: w={info['w']:.2f} -> ECI weegt {info['w']:.0%}, "
            f"markt {1 - info['w']:.0%}{extra}"
        )
    print(
        "Stabiliteitscheck: 'w_best_test' hoort in de buurt van 'w' te liggen. "
        "Wijkt het sterk af, dan is de klasse te klein of te ruizig."
    )

    print_table(
        "RELIABILITY OP TESTSET (gap ~ 0 na calibratie is goed)",
        reliability[reliability["source"].isin(["model", "calibrated"])],
        max_rows=120,
    )

    print_header("KLAAR")
    print(f"[export] gewichtenbestand: {weights_path} (versie {version})")
    print("De gewichten zijn nu BEVROREN. Wekelijks: --check. Herfit: op seizoensgrenzen.")

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, table in [
            ("calibration_class_mapping", mapping),
            ("calibration_fit_summary", summary),
            ("calibration_w_grid", wgrid_table),
            ("calibration_reliability", reliability),
        ]:
            path = CALIB_DIR / f"{name}_{stamp}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[export] {path}")

    return calib


# =====================================================================
# CHECK RUN (WEKELIJKSE MONITORING)
# =====================================================================

def run_check(
    source: str = DEFAULT_SOURCE,
    schema: str = DEFAULT_SCHEMA,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    since: str | None = None,
    export_csv: bool = False,
    refresh: bool = True,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Monitoring: leg de BEVROREN gewichten tegen nieuwe wedstrijden.

    Herfit niets. Rapporteert per klasse of de calibratie nog gezond is.
    ATTENTIE is een trigger voor een bewuste, geversioneerde herfit -
    niet voor een stille aanpassing.
    """
    print_header("BETMOBILE CALIBRATION CHECK")

    from prob_calibration import load_calibration

    calib = load_calibration(weights_path)
    print(
        f"gewichten: versie {calib['version']} | gefit op {calib['source']} "
        f"t/m {calib['data_window']['end']}"
    )

    class_calib = {
        "class_patterns": calib["class_patterns"],
        "default_class": calib["default_class"],
        "competition_class_map": calib.get("competition_class_map", {}),
    }

    if df is None:
        if refresh:
            try:
                refresh_source_views()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] refresh_source_views mislukt, ga door: {exc}")
        raw = load_match_frame(source, schema)
        df = prepare_match_frame(raw, class_calib=class_calib)
    else:
        df = prepare_match_frame(df, class_calib=class_calib)

    since_ts = pd.to_datetime(since) if since else pd.to_datetime(
        calib["data_window"]["end"]
    )
    part = df[df["date_dt"] > since_ts].copy()
    print(f"[check] wedstrijden na {since_ts.date()}: {len(part)}")

    if part.empty:
        print("Nog geen nieuwe wedstrijden om te checken.")
        return pd.DataFrame()

    w_by_class = {name: float(info["w"]) for name, info in calib["classes"].items()}
    default_w = w_by_class.get(
        calib["default_class"], float(calib["global"]["w"])
    )

    def ref_logloss(name: str) -> float | None:
        info = calib["classes"].get(name) if name != "ALL" else calib["global"]
        if info and info.get("logloss_test"):
            return float(info["logloss_test"]["calibrated"])
        return None

    rows = []
    groups = [("ALL", part)] + [
        (str(name), g) for name, g in part.groupby("calibration_class", observed=True)
    ]

    for name, g in groups:
        model_mat, market_mat, y = _matrices(g)
        if name == "ALL":
            w_arr = g["calibration_class"].map(w_by_class).fillna(default_w).to_numpy()
            w_shown = np.nan
        else:
            w_shown = w_by_class.get(name, default_w)
            w_arr = np.full(len(g), w_shown)

        cal_mat = blend_probs(model_mat, market_mat, w_arr)
        ll_cal = multiclass_logloss(cal_mat, y)
        ll_market = multiclass_logloss(market_mat, y)
        ll_model = multiclass_logloss(model_mat, y)

        ref = ref_logloss(name)
        delta_market = ll_cal - ll_market
        drift = (ll_cal - ref) if ref is not None else np.nan

        if len(g) < MIN_CHECK_MATCHES:
            status = "TE_WEINIG_DATA"
        elif delta_market > CHECK_MAX_DELTA_VS_MARKET or (
            ref is not None and drift > CHECK_MAX_DRIFT_VS_REF
        ):
            status = "ATTENTIE"
        else:
            status = "OK"

        rows.append(
            {
                "class": name,
                "n_matches": len(g),
                "w": w_shown,
                "ll_model": ll_model,
                "ll_market": ll_market,
                "ll_cal": ll_cal,
                "cal_vs_market": delta_market,
                "ref_ll_cal": ref,
                "drift_vs_ref": drift,
                "status": status,
            }
        )

    report = pd.DataFrame(rows)
    print_table("CHECK PER KLASSE", report)

    print_header("INTERPRETATIE")
    print(
        "OK             = bevroren gewichten doen nog wat ze bij de fit deden.\n"
        "TE_WEINIG_DATA = nog geen oordeel mogelijk; volgende week opnieuw.\n"
        "ATTENTIE       = calibratie slechter dan puur markt, of duidelijk\n"
        "                 verslechterd t.o.v. de fit-referentie. Dit is de\n"
        "                 trigger voor een BEWUSTE herfit met nieuw versienummer."
    )

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CALIB_DIR / f"calibration_check_{stamp}.csv"
        report.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[export] {path}")

    return report


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Betmobile calibration fit/check")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Monitor de bevroren gewichten tegen nieuwe data (geen herfit)",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Bronview met ALLE wedstrijden")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--since",
        default=None,
        help="Alleen voor --check: neem wedstrijden na deze datum (default: einde fit-window)",
    )
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--min-class-matches", type=int, default=MIN_CLASS_MATCHES)
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Refresh materialized views niet vooraf",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.check:
        run_check(
            source=args.source,
            schema=args.schema,
            weights_path=args.weights_path,
            since=args.since,
            export_csv=args.export_csv,
            refresh=not args.no_refresh,
        )
    else:
        run_fit(
            source=args.source,
            schema=args.schema,
            train_frac=args.train_frac,
            min_class_matches=args.min_class_matches,
            weights_path=args.weights_path,
            export_csv=args.export_csv,
            refresh=not args.no_refresh,
        )
