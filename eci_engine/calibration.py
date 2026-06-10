import pandas as pd
import numpy as np

from config import (
    CALIB_MIN_PICKS,
    CALIB_N_BANDS,
    CALIB_MIN_BETS_PER_BAND,
    CALIB_ROI_CLIP,
)


def build_calibration(hist: pd.DataFrame):
    if hist.empty or len(hist) < CALIB_MIN_PICKS:
        print(f"[CALIB] Te weinig historische picks ({len(hist)}), geen calibratie.")
        return None, None

    qs = np.linspace(0, 1, CALIB_N_BANDS + 1)
    quant = hist["rule_strength_adj"].quantile(qs).values
    edges = sorted(set(float(x) for x in quant))

    if len(edges) <= 2:
        print("[CALIB] Te weinig variatie, geen calibratie.")
        return None, None

    # Left-open intervals
    edges[0] = edges[0] - 1e-6
    edges[-1] = edges[-1] + 1e-6

    hist["band"] = pd.cut(
        hist["rule_strength_adj"], bins=edges, include_lowest=True, right=False
    )

    grp = hist.groupby("band", observed=True).agg(
        bets=("profit", "size"),
        profit=("profit", "sum")
    ).reset_index()

    grp["roi"] = grp["profit"] / grp["bets"].replace(0, np.nan)

    multipliers = []
    for _, row in grp.iterrows():
        if row["bets"] < CALIB_MIN_BETS_PER_BAND or pd.isna(row["roi"]):
            multipliers.append(1.0)
        else:
            roi = np.clip(row["roi"], -CALIB_ROI_CLIP, CALIB_ROI_CLIP)
            multipliers.append(1.0 + roi)

    grp["multiplier"] = multipliers

    print("\n[CALIB] Bands:")
    print(grp[["band", "bets", "profit", "roi", "multiplier"]].to_string(index=False))

    bands_meta = []
    for _, r in grp.iterrows():
        iv = r["band"]  # pandas.Interval
        bands_meta.append({
            "left": float(iv.left),
            "right": float(iv.right),
            "closed": str(iv.closed),
            "bets": int(r["bets"]),
            "profit": float(r["profit"]),
            "roi": (float(r["roi"]) if pd.notna(r["roi"]) else None),
            "multiplier": float(r["multiplier"]),
        })

    meta = {
        "n_hist": int(len(hist)),
        "bands": bands_meta
    }
    return meta, grp



# =====================================================================
# APPLY CALIBRATION
# =====================================================================
def apply_calibration(df: pd.DataFrame, calib_table):
    df = df.copy()

    def _ensure_numeric(col):
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _ensure_numeric("RuleStrengthAdj")
    _ensure_numeric("RuleStrengthAdj_Home")
    _ensure_numeric("RuleStrengthAdj_Away")

    if calib_table is None:
        df["RuleStrengthCalibrated"] = df["RuleStrengthAdj"]
        df["RuleStrengthCalibrated_Home"] = df.get("RuleStrengthAdj_Home")
        df["RuleStrengthCalibrated_Away"] = df.get("RuleStrengthAdj_Away")
        df["CalibMultiplier"] = 1.0
        df["CalibMultiplier_Home"] = 1.0
        df["CalibMultiplier_Away"] = 1.0
        return df

    band_index = pd.IntervalIndex.from_arrays(
        calib_table["band"].apply(lambda x: x.left),
        calib_table["band"].apply(lambda x: x.right),
        closed="left"
    )

    mult_map = dict(zip(band_index, calib_table["multiplier"]))

    def _apply_one(series: pd.Series):
        band = pd.cut(series, bins=band_index, include_lowest=True)
        mapped = band.map(mult_map)
        multiplier = mapped.astype("float64").fillna(1.0)
        calibrated = series * multiplier
        return calibrated, multiplier, band

    # Match-breed (oude hoofdveld, mag blijven bestaan)
    df["RuleStrengthCalibrated"], df["CalibMultiplier"], df["CalibBand"] = _apply_one(df["RuleStrengthAdj"])

    # Side-specifiek
    if "RuleStrengthAdj_Home" in df.columns:
        (
            df["RuleStrengthCalibrated_Home"],
            df["CalibMultiplier_Home"],
            df["CalibBand_Home"],
        ) = _apply_one(df["RuleStrengthAdj_Home"])

    if "RuleStrengthAdj_Away" in df.columns:
        (
            df["RuleStrengthCalibrated_Away"],
            df["CalibMultiplier_Away"],
            df["CalibBand_Away"],
        ) = _apply_one(df["RuleStrengthAdj_Away"])

    # Houd match-brede top ook beschikbaar
    if "RuleStrengthCalibrated_Home" in df.columns and "RuleStrengthCalibrated_Away" in df.columns:
        df["RuleStrengthCalibrated"] = df[["RuleStrengthCalibrated_Home", "RuleStrengthCalibrated_Away"]].max(axis=1)

    return df
