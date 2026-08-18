"""
historical_calibrated_compare.py

Generale repetitie voor de omschakeling naar probability-calibratie.

Draait de historische backtest (zelfde flow als research_backtest.py
--mode historical) twee keer op exact dezelfde wedstrijden:

  1. BASELINE   : rules op de rauwe ECI-kansen (huidige situatie)
  2. CALIBRATED : rules op de gekalibreerde kansen (nieuwe situatie)

Zo zie je VOOR de omschakeling precies wat de calibratie historisch met
het pickvolume en de ROI gedaan zou hebben. Dit verandert niets aan
productie of aan de database.

Vereist: een gefit gewichtenbestand (draai eerst fit_calibration.py).

Gebruik:
    python historical_calibrated_compare.py
    python historical_calibrated_compare.py --source betmobile_tuning_preko_mv --no-refresh
    python historical_calibrated_compare.py --export-csv
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from config import CALIBRATION_WEIGHTS_PATH, OUTPUT_DIR
from prob_calibration import load_calibration, calibrate_probs
from research_backtest import (
    DEFAULT_SOURCE,
    load_historical_backtest_data,
    prepare_backtest_frame,
    run_main_backtest,
    summarize_bets,
    print_header,
    print_result,
)

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def apply_prob_calibration_to_frame(df: pd.DataFrame, calib: dict) -> pd.DataFrame:
    """
    Vervang de kansen in een voorbereide backtest-frame door gekalibreerde
    kansen en reken de value scores opnieuw, zoals data_loader dat in
    productie doet.
    """
    df = calibrate_probs(
        df,
        calib,
        model_prob_cols=("home_win_pct", "draw_pct", "away_win_pct"),
    )

    df["Home Prob"] = df["prob_cal_home"]
    df["Draw Prob"] = df["prob_cal_draw"]
    df["Away Prob"] = df["prob_cal_away"]

    df["bet_home"] = df["odds_home"] * df["Home Prob"]
    df["bet_draw"] = df["odds_draw"] * df["Draw Prob"]
    df["bet_away"] = df["odds_away"] * df["Away Prob"]

    dropped = int(df["prob_cal_home"].isna().sum())
    if dropped:
        print(f"[calibratie] {dropped} wedstrijden zonder geldige kansen/odds.")
    return df


def summarize_run(label: str, picks: pd.DataFrame) -> None:
    print_header(f"RESULTAAT: {label}")
    if picks is None or picks.empty:
        print("Geen picks.")
        return
    for pick_type, part in picks.groupby("PickType", observed=True):
        print_result(summarize_bets(str(pick_type), part))
    print_result(summarize_bets(f"ALL {label}", picks))


def compare(base: pd.DataFrame, cal: pd.DataFrame) -> None:
    print_header("VERSCHIL BASELINE -> CALIBRATED")

    b = summarize_bets("baseline", base)
    c = summarize_bets("calibrated", cal)
    print(f"bets   : {b.bets} -> {c.bets}  ({c.bets - b.bets:+d})")
    print(f"profit : {b.profit:.2f} -> {c.profit:.2f}  ({c.profit - b.profit:+.2f})")
    print(f"roi    : {b.roi:.2%} -> {c.roi:.2%}  ({(c.roi - b.roi):+.2%})")
    print(f"hit    : {b.hitrate:.2%} -> {c.hitrate:.2%}")

    if base.empty or cal.empty:
        return

    key_cols = ["match_id", "Selection"]
    base_keys = set(map(tuple, base[key_cols].astype(str).values))
    cal_keys = set(map(tuple, cal[key_cols].astype(str).values))

    only_base = base[
        base[key_cols].astype(str).apply(tuple, axis=1).isin(base_keys - cal_keys)
    ]
    only_cal = cal[
        cal[key_cols].astype(str).apply(tuple, axis=1).isin(cal_keys - base_keys)
    ]

    print(f"\noverlap                : {len(base_keys & cal_keys)} picks")
    print(f"vervalt door calibratie: {len(only_base)} picks")
    print(f"nieuw door calibratie  : {len(only_cal)} picks")

    if not only_base.empty:
        gone = summarize_bets("weggevallen picks", only_base)
        print(
            f"\nDe weggevallen picks hadden historisch: bets={gone.bets}, "
            f"profit={gone.profit:.2f}, roi={gone.roi:.2%}"
        )
        print("(Negatieve roi hier = de calibratie filtert precies de slechte picks weg.)")

        if "competition" in only_base.columns:
            per_comp = (
                only_base.groupby("competition", observed=True)
                .agg(bets=("profit", "size"), profit=("profit", "sum"))
                .sort_values("bets", ascending=False)
                .head(15)
            )
            print("\nWeggevallen picks per competitie (top 15):")
            print(per_comp.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Historische backtest: raw vs gekalibreerd")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()

    print_header("GENERALE REPETITIE: RAW VS CALIBRATED")
    calib = load_calibration(CALIBRATION_WEIGHTS_PATH)
    print(f"gewichten: versie {calib['version']} (gefit op {calib['source']})")

    raw = load_historical_backtest_data(source=args.source, refresh_views=not args.no_refresh)
    df = prepare_backtest_frame(raw)
    if df.empty:
        print("Geen bruikbare historische data gevonden.")
        return

    picks_base = run_main_backtest(df)
    picks_cal = run_main_backtest(apply_prob_calibration_to_frame(df.copy(), calib))

    summarize_run("BASELINE (raw kansen)", picks_base)
    summarize_run("CALIBRATED", picks_cal)
    compare(picks_base, picks_cal)

    if args.export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, table in [("baseline", picks_base), ("calibrated", picks_cal)]:
            if table is not None and not table.empty:
                path = EXPORT_DIR / f"compare_{name}_picks_{stamp}.csv"
                table.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"[export] {path}")


if __name__ == "__main__":
    main()
