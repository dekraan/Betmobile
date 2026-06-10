import json
from datetime import datetime

import pandas as pd

from config import OUTPUT_DIR
from research_backtest_v6 import (
    load_picks_evaluated,
    prepare_picks_evaluated,
    summarize_bets,
    discover_segments,
)

SEGMENT_DIR = OUTPUT_DIR / "research"
SEGMENT_DIR.mkdir(parents=True, exist_ok=True)

STRONG_PATH = SEGMENT_DIR / "strong_segments.json"
DANGER_PATH = SEGMENT_DIR / "danger_segments.json"
REVIEW_PATH = SEGMENT_DIR / "segment_candidates_review.csv"


STRONG_MIN_BETS = 30
STRONG_MIN_ROI = 0.30
STRONG_MIN_ROI_VS_BASELINE = 0.20
STRONG_MIN_HITRATE = 0.65

DANGER_MIN_BETS = 30
DANGER_MAX_ROI = -0.10
DANGER_MAX_ROI_VS_BASELINE = -0.20


def parse_segment(segment_text: str) -> dict:
    """
    Zet:
    'prob_bucket=65-70% | odds_bucket=1.6-1.8'
    om naar:
    {'prob_bucket': '65-70%', 'odds_bucket': '1.6-1.8'}
    """
    conditions = {}

    for part in str(segment_text).split("|"):
        part = part.strip()
        if "=" not in part:
            continue

        key, value = part.split("=", 1)
        conditions[key.strip()] = value.strip()

    return conditions


def make_segment_name(prefix: str, row: pd.Series) -> str:
    raw = str(row["segment"])
    clean = (
        raw.replace("=", "_")
           .replace(" | ", "__")
           .replace(" ", "_")
           .replace("%", "pct")
           .replace("/", "_")
           .replace("+", "plus")
           .replace("<", "lt")
           .replace(">", "gt")
           .replace(".", "_")
           .replace("-", "_")
           .replace(":", "")
           .lower()
    )
    return f"{prefix}_{clean}"[:120]


def row_to_segment(prefix: str, row: pd.Series) -> dict:
    return {
        "name": make_segment_name(prefix, row),
        "conditions": parse_segment(row["segment"]),
        "meta": {
            "dimensions": row.get("dimensions"),
            "bets": int(row.get("bets", 0)),
            "wins": int(row.get("wins", 0)),
            "profit": float(row.get("profit", 0.0)),
            "roi": float(row.get("roi", 0.0)),
            "roi_vs_baseline": float(row.get("roi_vs_baseline", 0.0)),
            "hitrate": float(row.get("hitrate", 0.0)),
            "avg_odds": float(row.get("avg_odds", 0.0)),
            "avg_prob_selected": float(row.get("avg_prob_selected", 0.0)),
            "avg_value": float(row.get("avg_value", 0.0)),
            "avg_strength": float(row.get("avg_strength", 0.0)),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def dedupe_segments(segments: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for seg in segments:
        key = tuple(sorted(seg["conditions"].items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)

    return out


def main():
    raw = load_picks_evaluated()
    df = prepare_picks_evaluated(raw)

    if df.empty:
        print("Geen settled picks gevonden.")
        return

    baseline_roi = summarize_bets("baseline", df).roi
    best_segments, worst_segments, interesting_segments = discover_segments(
        df,
        baseline_roi=baseline_roi,
        min_bets=20,
    )

    candidates = pd.concat(
        [
            best_segments.assign(candidate_type="BEST"),
            worst_segments.assign(candidate_type="WORST"),
            interesting_segments.assign(candidate_type="INTERESTING"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["dimensions", "segment"])

    candidates.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    strong_df = candidates[
        (candidates["bets"] >= STRONG_MIN_BETS)
        & (candidates["roi"] >= STRONG_MIN_ROI)
        & (candidates["roi_vs_baseline"] >= STRONG_MIN_ROI_VS_BASELINE)
        & (candidates["hitrate"] >= STRONG_MIN_HITRATE)
    ].copy()

    danger_df = candidates[
        (candidates["bets"] >= DANGER_MIN_BETS)
        & (candidates["roi"] <= DANGER_MAX_ROI)
        & (candidates["roi_vs_baseline"] <= DANGER_MAX_ROI_VS_BASELINE)
    ].copy()

    strong_segments = [
        row_to_segment("strong", row)
        for _, row in strong_df.sort_values(["roi_vs_baseline", "bets"], ascending=[False, False]).iterrows()
    ]

    danger_segments = [
        row_to_segment("danger", row)
        for _, row in danger_df.sort_values(["roi_vs_baseline", "bets"], ascending=[True, False]).iterrows()
    ]

    strong_segments = dedupe_segments(strong_segments)
    danger_segments = dedupe_segments(danger_segments)

    STRONG_PATH.write_text(
        json.dumps({"segments": strong_segments}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    DANGER_PATH.write_text(
        json.dumps({"segments": danger_segments}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Baseline ROI: {baseline_roi:.2%}")
    print(f"Review CSV: {REVIEW_PATH}")
    print(f"Strong segments: {len(strong_segments)} → {STRONG_PATH}")
    print(f"Danger segments: {len(danger_segments)} → {DANGER_PATH}")


if __name__ == "__main__":
    main()