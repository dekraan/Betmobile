"""
strength_analysis.py

Rapport over settled picks in public.picks_evaluated.

Gebruik:
    python strength_analysis.py

Export:
    output/research/strength_analysis_YYYYMMDD.xlsx

Tabs:
    Summary, By tier, By strength, By pick_type, By segment flags, By time

Tiering:
    pick_tier leeg/ontbrekend → UNKNOWN (geen historische herclassificatie).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from config import OUTPUT_DIR
from db import db_engine
from research_backtest import prepare_picks_evaluated, summarize_bets


EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

TIER_ORDER = ["A+", "A", "A-", "B", "C", "X", "UNKNOWN"]
STRENGTH_ORDER = ["<1", "1-1.5", "1.5-2", "2-3", "3+", "UNKNOWN"]


def load_strength_analysis_picks() -> pd.DataFrame:
    """Laad settled picks inclusief classificatievelden."""
    q = text("""
        SELECT
            run_id,
            match_id,
            competition,
            date,
            home_team,
            away_team,
            odds_home,
            odds_draw,
            odds_away,
            prob_home,
            prob_draw,
            prob_away,
            bet_home,
            bet_draw,
            bet_away,
            edge_h,
            edge_d,
            edge_a,
            rule_a,
            rule_b,
            rule_c,
            pick_reason,
            rule_strength,
            rule_strength_adj,
            away_drift_pct,
            home_drift_pct,
            n_snapshots,
            hours_stale,
            selection,
            score,
            result,
            outcome,
            settled_at,
            date_ts,
            rule_passed,
            rule_reason,
            pick_type,
            value_edge,
            prob_edge,
            drift_score,
            snapshot_count,
            strength_bucket,
            rating_gap,
            drift_range,
            pick_tier,
            pick_stars,
            sector_tags,
            danger_tags,
            classification_reason,
            passes_danger_combo_v2,
            passes_danger_combo_v2_no_longshots
        FROM public.picks_evaluated
        WHERE selection IS NOT NULL
          AND outcome IN ('WIN', 'LOSS')
    """)

    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    print(f"[strength_analysis] settled rows loaded: {len(df)}")
    return df


def normalize_pick_tier(series: pd.Series) -> pd.Series:
    tier = series.fillna("").astype(str).str.strip()
    tier = tier.replace("", "UNKNOWN")
    return tier


def add_segment_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sector = df.get("sector_tags", pd.Series("", index=df.index)).fillna("").astype(str)
    danger = df.get("danger_tags", pd.Series("", index=df.index)).fillna("").astype(str)

    df["dynamic_strong_present"] = np.where(
        sector.str.contains("dynamic_strong:", regex=False),
        "YES",
        "NO",
    )
    df["dynamic_danger_present"] = np.where(
        danger.str.contains("dynamic_danger:", regex=False),
        "YES",
        "NO",
    )

    combo = np.select(
        [
            (df["dynamic_strong_present"] == "YES") & (df["dynamic_danger_present"] == "YES"),
            (df["dynamic_strong_present"] == "YES") & (df["dynamic_danger_present"] == "NO"),
            (df["dynamic_strong_present"] == "NO") & (df["dynamic_danger_present"] == "YES"),
        ],
        ["strong_and_danger", "strong_only", "danger_only"],
        default="neither",
    )
    df["segment_combo"] = combo
    return df


def summarize_group(
    df: pd.DataFrame,
    group_cols: list[str],
    baseline_roi: float,
    min_bets: int = 1,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = (
        df.groupby(group_cols, observed=True, dropna=False)
        .agg(
            bets=("profit", "size"),
            wins=("won", "sum"),
            profit=("profit", "sum"),
            avg_odds=("selected_odds", "mean"),
            avg_strength=("rule_strength_adj", "mean"),
            avg_prob=("prob_selected", "mean"),
        )
        .reset_index()
    )

    out["losses"] = out["bets"] - out["wins"]
    out["roi"] = out["profit"] / out["bets"]
    out["hitrate"] = out["wins"] / out["bets"]
    out["roi_vs_baseline"] = out["roi"] - baseline_roi
    out = out[out["bets"] >= min_bets].copy()
    return out


def sort_by_custom_order(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    order_map = {value: idx for idx, value in enumerate(order)}
    df = df.copy()
    df["_sort"] = df[col].map(order_map).fillna(len(order))
    df = df.sort_values(["_sort", col]).drop(columns="_sort")
    return df


def build_summary_sheet(df: pd.DataFrame, baseline_roi: float) -> pd.DataFrame:
    overall = summarize_bets("ALL_SETTLED_PICKS", df)
    known_tier = df[df["pick_tier"] != "UNKNOWN"]
    tiered = summarize_bets("PICKS_WITH_KNOWN_TIER", known_tier)

    first_date = pd.to_datetime(df["date"], errors="coerce").min()
    last_date = pd.to_datetime(df["date"], errors="coerce").max()
    tier_from = (
        pd.to_datetime(known_tier["date"], errors="coerce").min()
        if not known_tier.empty
        else pd.NaT
    )

    rows = [
        {"metric": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"metric": "total_settled_picks", "value": overall.bets},
        {"metric": "first_pick_date", "value": first_date.date() if pd.notna(first_date) else ""},
        {"metric": "last_pick_date", "value": last_date.date() if pd.notna(last_date) else ""},
        {"metric": "baseline_roi", "value": round(baseline_roi, 4)},
        {"metric": "baseline_hitrate", "value": round(overall.hitrate, 4)},
        {"metric": "baseline_profit", "value": round(overall.profit, 2)},
        {"metric": "picks_with_known_tier", "value": len(known_tier)},
        {"metric": "picks_unknown_tier", "value": int((df["pick_tier"] == "UNKNOWN").sum())},
        {
            "metric": "tier_data_from",
            "value": tier_from.date() if pd.notna(tier_from) else "",
        },
        {"metric": "known_tier_roi", "value": round(tiered.roi, 4)},
        {"metric": "known_tier_hitrate", "value": round(tiered.hitrate, 4)},
        {"metric": "note", "value": "Tier-tab: UNKNOWN = picks vóór live tiering. Geen herclassificatie."},
    ]

    summary = pd.DataFrame(rows)

    by_type = summarize_group(df, ["pick_type"], baseline_roi, min_bets=1)
    if not by_type.empty:
        by_type = by_type.rename(columns={"pick_type": "pick_type_summary"})
        by_type.insert(0, "section", "pick_type_overview")
        summary = pd.concat([summary, pd.DataFrame([{}]), by_type], ignore_index=True, sort=False)

    return summary


def build_by_time_sheet(df: pd.DataFrame, baseline_roi: float) -> pd.DataFrame:
    by_month = summarize_group(df, ["month"], baseline_roi, min_bets=1)
    if not by_month.empty:
        by_month.insert(0, "time_dimension", "month")
        by_month = by_month.rename(columns={"month": "time_bucket"})

    by_phase = summarize_group(df, ["season_phase"], baseline_roi, min_bets=1)
    if not by_phase.empty:
        by_phase.insert(0, "time_dimension", "season_phase")
        by_phase = by_phase.rename(columns={"season_phase": "time_bucket"})

    parts = [part for part in [by_month, by_phase] if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for col in ["profit", "roi", "hitrate", "roi_vs_baseline", "avg_odds", "avg_strength", "avg_prob"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def export_strength_analysis(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d")
        output_path = EXPORT_DIR / f"strength_analysis_{stamp}.xlsx"

    baseline = summarize_bets("baseline", df)
    baseline_roi = baseline.roi

    by_tier = sort_by_custom_order(
        summarize_group(df, ["pick_tier"], baseline_roi, min_bets=1),
        "pick_tier",
        TIER_ORDER,
    )
    by_strength = sort_by_custom_order(
        summarize_group(df, ["strength_bucket"], baseline_roi, min_bets=1),
        "strength_bucket",
        STRENGTH_ORDER,
    )
    by_pick_type = summarize_group(df, ["pick_type"], baseline_roi, min_bets=1)
    by_segment_flags = summarize_group(
        df,
        ["dynamic_strong_present", "dynamic_danger_present", "segment_combo"],
        baseline_roi,
        min_bets=1,
    )
    by_strength_pick_type = summarize_group(
        df,
        ["strength_bucket", "pick_type"],
        baseline_roi,
        min_bets=1,
    )
    by_odds_strength = summarize_group(
        df,
        ["odds_bucket", "strength_bucket"],
        baseline_roi,
        min_bets=1,
    )
    by_odds_strength = sort_by_custom_order(
        by_odds_strength,
        "strength_bucket",
        STRENGTH_ORDER,
    )
    by_time = build_by_time_sheet(df, baseline_roi)
    summary = build_summary_sheet(df, baseline_roi)
    by_strength_pick_type = sort_by_custom_order(
        by_strength_pick_type,
        "strength_bucket",
        STRENGTH_ORDER,
    )
    sheets = {
        "Summary": round_numeric_columns(summary),
        "By tier": round_numeric_columns(by_tier),
        "By strength": round_numeric_columns(by_strength),
        "By strength x type": round_numeric_columns(by_strength_pick_type),
        "By odds x strength": round_numeric_columns(by_odds_strength),
        "By pick_type": round_numeric_columns(by_pick_type),
        "By segment flags": round_numeric_columns(by_segment_flags),
        "By time": round_numeric_columns(by_time),
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_path


def print_console_summary(df: pd.DataFrame, baseline_roi: float) -> None:
    print("\n=== STRENGTH ANALYSIS ===")
    print(f"Settled picks: {len(df)}")
    print(f"Baseline ROI: {baseline_roi:.2%}")

    known_tier = df[df["pick_tier"] != "UNKNOWN"]
    print(f"Picks with known tier: {len(known_tier)}")
    if not known_tier.empty:
        tier_from = pd.to_datetime(known_tier["date"], errors="coerce").min()
        print(f"Tier data from: {tier_from.date()}")

    by_tier = sort_by_custom_order(
        summarize_group(df, ["pick_tier"], baseline_roi, min_bets=1),
        "pick_tier",
        TIER_ORDER,
    )
    if not by_tier.empty:
        print("\n--- By tier ---")
        print(
            by_tier[
                ["pick_tier", "bets", "profit", "roi", "hitrate", "roi_vs_baseline"]
            ].to_string(index=False)
        )


def main() -> None:
    raw = load_strength_analysis_picks()
    if raw.empty:
        print("Geen settled picks gevonden in public.picks_evaluated.")
        return

    df = prepare_picks_evaluated(raw)
    df["pick_tier"] = normalize_pick_tier(df.get("pick_tier", pd.Series(dtype=object)))
    df["pick_type"] = df["pick_type"].fillna("UNKNOWN").replace("", "UNKNOWN")
    df = add_segment_flags(df)

    baseline_roi = summarize_bets("baseline", df).roi
    print_console_summary(df, baseline_roi)

    output_path = export_strength_analysis(df)
    print(f"\nExcel export: {output_path}")


if __name__ == "__main__":
    main()
