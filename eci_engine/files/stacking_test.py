"""
stacking_test.py

De laatste vooraf vastgelegde test van de vraag:
    "Voegt ECI voorspellende informatie toe bovenop de markt?"

PROTOCOL (vastgelegd 2026-08-19, voor het zien van de uitslag):
1. Lineaire blend        -> gedaan (fit_calibration.py) -> w = 0, geen bijdrage.
2. Conditionele tabel    -> deze run: markt-bucket x (ECI - markt)-bucket
                            tegen werkelijke hitrate. Beschrijvend, geen model.
3. Logistische stacking  -> deze run: markt als baseline, ECI als feature,
                            zelfde datumsplitsing als de calibratie-fit.
PRIMAIR CRITERIUM: test-log-loss van de stacking moet beter zijn dan de kale
marktbaseline, met een 95% bootstrap-interval dat nul uitsluit.
ALS DAT NIET LUKT: ECI wordt niet langer als extra signaal behandeld en de
verzamelfase loopt door. Geen extra modellen, geen nieuwe lagen.

Gebruik:
    python stacking_test.py
    python stacking_test.py --export-csv --no-refresh
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from fit_calibration import (
    DEFAULT_SCHEMA,
    DEFAULT_SOURCE,
    TRAIN_FRAC,
    PICK_ZONE_MIN_GAP,
    PICK_ZONE_MIN_ODDS,
    PICK_ZONE_MAX_ODDS,
    load_match_frame,
    prepare_match_frame,
    multiclass_logloss,
    multiclass_brier,
    print_header,
    print_table,
)
from db import refresh_source_views

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_BUCKETS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
GAP_BUCKETS = [-1.0, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1.0]
GAP_LABELS = ["<-10pp", "-10..-5", "-5..-2", "-2..+2", "+2..+5", "+5..+10", ">+10pp"]

BOOTSTRAP_N = 2000
RNG = np.random.default_rng(42)


# =====================================================================
# HULPFUNCTIES
# =====================================================================

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Elke wedstrijd wordt drie observaties (H/D/A) met binaire uitkomst."""
    frames = []
    for i, side in enumerate(["home", "draw", "away"]):
        frames.append(
            pd.DataFrame(
                {
                    "match_idx": df.index,
                    "side": side,
                    "p_mkt": df[f"mkt_{side}"].to_numpy(float),
                    "p_eci": df[f"mdl_{side}"].to_numpy(float),
                    "hit": (df["y_idx"].to_numpy(int) == i).astype(float),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    long["gap"] = long["p_eci"] - long["p_mkt"]
    return long


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logistic_irls(X: np.ndarray, y: np.ndarray, iters: int = 30, ridge: float = 1e-6) -> np.ndarray:
    """Kleine, dependency-vrije logistische regressie (IRLS)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def stacked_probs(df: pd.DataFrame, beta: np.ndarray, features_fn) -> np.ndarray:
    """Per outcome logistisch voorspellen, daarna per wedstrijd normaliseren."""
    cols = []
    for side in ["home", "draw", "away"]:
        X = features_fn(df, side)
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        cols.append(p)
    mat = np.column_stack(cols)
    return mat / mat.sum(axis=1, keepdims=True)


def features_primary(df: pd.DataFrame, side: str) -> np.ndarray:
    """PRIMAIR (vooraf vastgelegd): intercept + logit(markt) + logit(ECI)."""
    return np.column_stack(
        [
            np.ones(len(df)),
            logit(df[f"mkt_{side}"].to_numpy(float)),
            logit(df[f"mdl_{side}"].to_numpy(float)),
        ]
    )


def features_secondary(df: pd.DataFrame, side: str) -> np.ndarray:
    """SECUNDAIR: idem + aparte intercepts per uitkomsttype (H/D/A)."""
    d_home = np.full(len(df), 1.0 if side == "home" else 0.0)
    d_away = np.full(len(df), 1.0 if side == "away" else 0.0)
    return np.column_stack(
        [
            np.ones(len(df)),
            d_home,
            d_away,
            logit(df[f"mkt_{side}"].to_numpy(float)),
            logit(df[f"mdl_{side}"].to_numpy(float)),
        ]
    )


def fit_stacking(train: pd.DataFrame, features_fn) -> np.ndarray:
    Xs, ys = [], []
    for i, side in enumerate(["home", "draw", "away"]):
        Xs.append(features_fn(train, side))
        ys.append((train["y_idx"].to_numpy(int) == i).astype(float))
    return fit_logistic_irls(np.vstack(Xs), np.concatenate(ys))


def per_match_logloss(prob_mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = prob_mat[np.arange(len(y)), y]
    return -np.log(np.clip(p, 1e-9, None))


# =====================================================================
# DEEL 1: CONDITIONELE TABEL
# =====================================================================

def conditional_table(long: pd.DataFrame):
    long = long.copy()
    long["mkt_bucket"] = pd.cut(long["p_mkt"], bins=MARKET_BUCKETS, include_lowest=True)
    long["gap_bucket"] = pd.cut(long["gap"], bins=GAP_BUCKETS, labels=GAP_LABELS)

    grp = (
        long.groupby(["mkt_bucket", "gap_bucket"], observed=True)
        .agg(n=("hit", "size"), hitrate=("hit", "mean"), avg_mkt=("p_mkt", "mean"))
        .reset_index()
    )
    grp["lift"] = grp["hitrate"] - grp["avg_mkt"]

    piv_n = grp.pivot(index="mkt_bucket", columns="gap_bucket", values="n")
    piv_hit = grp.pivot(index="mkt_bucket", columns="gap_bucket", values="hitrate")
    piv_lift = grp.pivot(index="mkt_bucket", columns="gap_bucket", values="lift")

    # Samenvatting per gap-kolom over alle marktrijen heen.
    per_gap = (
        long.groupby("gap_bucket", observed=True)
        .agg(n=("hit", "size"), hitrate=("hit", "mean"), avg_mkt=("p_mkt", "mean"))
        .reset_index()
    )
    per_gap["lift"] = per_gap["hitrate"] - per_gap["avg_mkt"]

    # Slope beta: (hit - p_mkt) ~ beta * (p_eci - p_mkt).
    # beta = 0: ECI-afwijking zegt niets. beta = 1: ECI heeft volledig gelijk
    # over zijn afwijking. (De lineaire blend-w is hiervan de broertjesmaat.)
    x = long["gap"].to_numpy(float)
    r = (long["hit"] - long["p_mkt"]).to_numpy(float)
    beta = float(np.sum(x * r) / np.sum(x * x))
    resid = r - beta * x
    se = float(np.sqrt(np.sum(resid**2) / (len(x) - 1) / np.sum(x * x)))

    return piv_n, piv_hit, piv_lift, per_gap, beta, se


# =====================================================================
# HOOFDPROGRAMMA
# =====================================================================

def run(source: str, schema: str, train_frac: float, refresh: bool,
        export_csv: bool, df: pd.DataFrame | None = None) -> dict:
    print_header("STACKING TEST: VOEGT ECI IETS TOE BOVENOP DE MARKT?")
    print(
        "Vooraf vastgelegd criterium: stacking-log-loss op de testset beter dan\n"
        "de kale marktbaseline, met 95% bootstrap-interval dat nul uitsluit.\n"
        "Zo niet: ECI wordt niet langer als extra signaal behandeld."
    )

    if df is None:
        if refresh:
            try:
                refresh_source_views()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] refresh_source_views mislukt, ga door: {exc}")
        df = prepare_match_frame(load_match_frame(source, schema))
    else:
        df = prepare_match_frame(df)

    df = df.reset_index(drop=True)
    split_date = df["date_dt"].quantile(train_frac)
    train = df[df["date_dt"] <= split_date]
    test = df[df["date_dt"] > split_date]
    print(f"[split] {pd.Timestamp(split_date).date()} | train={len(train)} | test={len(test)}")
    if len(test) < 300:
        raise RuntimeError("Testset te klein voor een betrouwbaar oordeel.")

    # ---------- DEEL 1: conditionele tabel (volledige set, beschrijvend) ----------
    long = to_long(df)
    piv_n, piv_hit, piv_lift, per_gap, beta, se = conditional_table(long)

    print_table("CONDITIONEEL: AANTAL WAARNEMINGEN (markt-bucket x ECI-minus-markt)", piv_n.reset_index())
    print_table("CONDITIONEEL: WERKELIJKE HITRATE", piv_hit.reset_index())
    print_table("CONDITIONEEL: LIFT (hitrate minus marktclaim; >0 rechts = ECI weet iets)", piv_lift.reset_index())
    print_table("CONDITIONEEL: SAMENVATTING PER GAP-KOLOM", per_gap)
    print_header("SLOPE-DIAGNOSE")
    print(
        f"beta = {beta:+.4f} (SE {se:.4f})\n"
        "beta ~ 0: ECI's afwijking van de markt voorspelt niets extra's.\n"
        "beta ~ 1: ECI's afwijking klopt volledig. (Ter referentie: de blend vond w = 0.)"
    )

    # ---------- DEEL 2: stacking op dezelfde split ----------
    y_test = test["y_idx"].to_numpy(int)
    mkt_test = test[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy(float)
    eci_test = test[["mdl_home", "mdl_draw", "mdl_away"]].to_numpy(float)

    beta_p = fit_stacking(train, features_primary)
    beta_s = fit_stacking(train, features_secondary)
    stack_p = stacked_probs(test, beta_p, features_primary)
    stack_s = stacked_probs(test, beta_s, features_secondary)

    rows = []
    for name, mat in [
        ("markt (baseline)", mkt_test),
        ("ECI alleen", eci_test),
        ("stacking PRIMAIR (markt+ECI logit)", stack_p),
        ("stacking secundair (+H/D/A intercepts)", stack_s),
    ]:
        rows.append(
            {
                "model": name,
                "logloss_test": multiclass_logloss(mat, y_test),
                "brier_test": multiclass_brier(mat, y_test),
            }
        )
    results = pd.DataFrame(rows)
    base_ll = results.loc[results["model"] == "markt (baseline)", "logloss_test"].iloc[0]
    results["vs_markt"] = results["logloss_test"] - base_ll
    print_table("STACKING: TEST LOG LOSS (lager = beter)", results)
    print(f"[coef] primair (intercept, logit_mkt, logit_eci): {np.round(beta_p, 4).tolist()}")

    # ---------- Bootstrap-oordeel (vooraf vastgelegd) ----------
    lm = per_match_logloss(mkt_test, y_test)
    ls = per_match_logloss(stack_p, y_test)
    diff = ls - lm
    idx = RNG.integers(0, len(diff), size=(BOOTSTRAP_N, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print_header("OORDEEL (vooraf vastgelegd criterium)")
    print(
        f"gemiddeld verschil stacking - markt: {diff.mean():+.5f} "
        f"(95% CI [{lo:+.5f}, {hi:+.5f}])"
    )
    beaten = diff.mean() < 0 and hi < 0
    verdict = (
        "SIGNAAL GEVONDEN: stacking verslaat de markt significant. ECI bevat\n"
        "conditionele informatie; volgende stap is bespreken hoe we dit inzetten."
        if beaten
        else "GEEN SIGNAAL: stacking verslaat de marktbaseline niet.\n"
        "Afspraak treedt in werking: ECI wordt niet langer als extra signaal\n"
        "behandeld. De verzamelfase (bevroren regels, out-of-sample paneel,\n"
        "CLV) loopt door; geen nieuwe modellagen."
    )
    print(verdict)

    # ---------- Pick-zone slice (alleen rapportage, geen refit) ----------
    if {"home_rating", "away_rating"} <= set(test.columns):
        hr = pd.to_numeric(test["home_rating"], errors="coerce")
        ar = pd.to_numeric(test["away_rating"], errors="coerce")
        fav = test[["odds_home", "odds_away"]].min(axis=1)
        pz = (hr - ar).abs() >= PICK_ZONE_MIN_GAP
        pz &= fav.between(PICK_ZONE_MIN_ODDS, PICK_ZONE_MAX_ODDS)
        if int(pz.sum()) >= 100:
            sel = pz.to_numpy()
            print_header(f"PICK-ZONE SLICE VAN DE TESTSET (n={int(pz.sum())})")
            print(
                f"markt   : {multiclass_logloss(mkt_test[sel], y_test[sel]):.4f}\n"
                f"stacking: {multiclass_logloss(stack_p[sel], y_test[sel]):.4f}"
            )

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, table in [
            ("stacking_conditional_n", piv_n.reset_index()),
            ("stacking_conditional_hitrate", piv_hit.reset_index()),
            ("stacking_conditional_lift", piv_lift.reset_index()),
            ("stacking_per_gap", per_gap),
            ("stacking_results", results),
        ]:
            path = EXPORT_DIR / f"{name}_{stamp}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[export] {path}")

    return {
        "beta_conditional": beta,
        "beaten": bool(beaten),
        "diff_mean": float(diff.mean()),
        "ci": (float(lo), float(hi)),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ECI-als-feature: laatste vastgelegde test")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()
    run(
        source=args.source,
        schema=args.schema,
        train_frac=args.train_frac,
        refresh=not args.no_refresh,
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    main()