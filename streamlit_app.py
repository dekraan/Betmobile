"""
Betmobile Streamlit Cockpit v0.8

Plaats dit bestand in:
C:/Users/Gebruiker/Documents/Betmobile/streamlit_app.py

Starten:
    cd C:/Users/Gebruiker/Documents/Betmobile
    streamlit run streamlit_app_v08.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# -----------------------------------------------------------------------------
# Projectpaden
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ECI_ENGINE_DIR = BASE_DIR / "eci_engine"

RUN_MODEL_PATH = ECI_ENGINE_DIR / "run_model.py"

SNAPSHOT_PATH = BASE_DIR / "run_snapshot.py"
SETTLE_PATH = BASE_DIR / "settle_picks.py"
MAINTENANCE_PATH = BASE_DIR / "run_daily_maintenance.py"

SCRAPER_PATH = BASE_DIR / "eci_scraper.py"

# Zorg dat imports uit eci_engine werken, net als bij run_model.py.
for p in [BASE_DIR, ECI_ENGINE_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from config import (  # type: ignore
        DB_DSN,
        ECI_RULE_PARAMS,
        MIN_STRENGTH,
        USE_CUTOFF_FEATURES,
        DRIFT_SUPPORT_THRESHOLD,
        DRIFT_OPPOSE_THRESHOLD,
        DRIFT_SUPPORT_BONUS,
        DRIFT_OPPOSE_PENALTY,
        SNAP_BONUS_THRESHOLD,
        SNAP_BONUS,
        RANGE_PENALTY_THRESHOLD,
        RANGE_PENALTY,
        ENABLE_SECONDARY_PICKS,
        SECONDARY_ALLOWED_FAIL,
        SECONDARY_VALUE_TOLERANCE,
        SECONDARY_MIN_STRENGTH,
        SECONDARY_MIN_PROB,
    )
except Exception:
    # Fallback zodat de app ten minste kan starten als config-import faalt.
    DB_DSN = os.getenv(
        "BETMOBILE_DB_DSN",
        "postgresql+psycopg2://postgres:300500@localhost:5432/Betmobile",
    )
    ECI_RULE_PARAMS = {
        "min_prob": 0.52,
        "min_value": 1.04,
        "min_rating_gap": 0,
        "min_odds": 1.4,
        "max_odds": 4.0,
        "min_snapshots": 7,
        "min_drift_abs": 0,
    }
    MIN_STRENGTH = 1.5
    USE_CUTOFF_FEATURES = False
    DRIFT_SUPPORT_THRESHOLD = -0.03
    DRIFT_OPPOSE_THRESHOLD = 0.03
    DRIFT_SUPPORT_BONUS = 0.10
    DRIFT_OPPOSE_PENALTY = 0.10
    SNAP_BONUS_THRESHOLD = 15
    SNAP_BONUS = 0.05
    RANGE_PENALTY_THRESHOLD = 0.50
    RANGE_PENALTY = 0.05
    ENABLE_SECONDARY_PICKS = True
    SECONDARY_ALLOWED_FAIL = "value"
    SECONDARY_VALUE_TOLERANCE = 0.04
    SECONDARY_MIN_STRENGTH = 1.8
    SECONDARY_MIN_PROB = 0.50

st.set_page_config(page_title="Betmobile Cockpit", page_icon="⚽", layout="wide")


st.markdown(
    """
    <style>
    .bm-card-title {font-size: 1.15rem; font-weight: 700; margin-bottom: .15rem;}
    .bm-muted {color: #6b7280; font-size: .88rem;}
    .bm-chip {display:inline-block; padding: .18rem .48rem; border-radius: 999px; margin: .08rem .18rem .08rem 0; font-size: .78rem; font-weight: 600; border: 1px solid rgba(49,51,63,.18);}
    .bm-chip-good {background: rgba(46, 160, 67, .12);}
    .bm-chip-warn {background: rgba(210, 153, 34, .16);}
    .bm-chip-bad {background: rgba(248, 81, 73, .12);}
    .bm-chip-neutral {background: rgba(120, 120, 120, .10);}
    .bm-small {font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------

@st.cache_resource
def get_engine():
    return create_engine(DB_DSN)


@st.cache_data(ttl=60)
def query_df(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


@st.cache_data(ttl=300)
def table_columns(table_name: str, schema: str = "public") -> pd.DataFrame:
    return query_df(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table_name
        ORDER BY ordinal_position;
        """,
        {"schema": schema, "table_name": table_name},
    )


def run_python_script(script_path: Path, cwd: Path) -> tuple[int, str, str]:
    if not script_path.exists():
        return 1, "", f"Script niet gevonden: {script_path}"

    p = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=900,
    )
    return p.returncode, p.stdout, p.stderr


def file_mtime(path: Path) -> str:
    if not path.exists():
        return "Niet gevonden"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

def file_age_minutes(path: Path) -> str:
    if not path.exists():
        return "?"

    age = (
        datetime.now().timestamp()
        - path.stat().st_mtime
    ) / 60

    return f"{age:.0f} min"

def age_text(dt_value):
    if dt_value is None or pd.isna(dt_value):
        return "?"

    try:
        dt = pd.to_datetime(dt_value, errors="coerce")

        if pd.isna(dt):
            return "?"

        if dt.tzinfo is not None:
            dt = dt.tz_convert(None)

        diff = datetime.now() - dt.to_pydatetime()
        minutes = int(diff.total_seconds() / 60)

        if minutes < 0:
            return "net"

        if minutes < 60:
            return f"{minutes} min geleden"

        hours = minutes // 60

        if hours < 24:
            return f"{hours} uur geleden"

        days = hours // 24
        return f"{days} dag(en) geleden"

    except Exception:
        return "?"

def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["selected_prob", "selected_value", "selected_drift_pct", "hitrate", "roi", "probability", "value_score", "selected_prob", "selected_value"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    for col in ["selected_odds", "rule_strength", "rule_strength_adj", "rating_gap", "strength", "single_fail_margin", "single_fail_raw_strength", "single_fail_adj_strength", "single_fail_calibrated_strength", "odds"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


def show_df(df: pd.DataFrame, empty_text: str) -> None:
    if df.empty:
        st.info(empty_text)
    else:
        st.dataframe(prepare_display_df(df), width="stretch", hide_index=True)


def metric_value(x: Any) -> Any:
    if x is None or pd.isna(x):
        return "—"
    return x


def fmt_pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


def fmt_num(x: Any, ndigits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def tier_badge(tier: Any, stars: Any) -> str:
    tier_txt = str(tier).strip() if tier is not None and not pd.isna(tier) and str(tier).strip() else "?"
    try:
        n = int(stars) if stars is not None and not pd.isna(stars) else 0
    except Exception:
        n = 0
    star_txt = "⭐" * max(0, min(n, 5))
    return f"{tier_txt} {star_txt}".strip()


def split_tags(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    raw = str(value).replace(";", ",")
    return [x.strip() for x in raw.split(",") if x.strip()]


def selected_strength(row: pd.Series) -> Any:
    for col in ["rule_strength_adj", "rule_strength"]:
        if col in row and row.get(col) is not None and not pd.isna(row.get(col)):
            return row.get(col)
    return None


def tier_class(tier: Any, danger_tags: list[str]) -> str:
    t = str(tier or "").strip().upper()
    if t == "X" or len(danger_tags) >= 2:
        return "bad"
    if t in {"A+", "A"}:
        return "good"
    if t in {"A-", "B", "C"} or danger_tags:
        return "warn"
    return "neutral"


def chip(text: str, kind: str = "neutral") -> str:
    safe = str(text).replace("<", "&lt;").replace(">", "&gt;")
    return f'<span class="bm-chip bm-chip-{kind}">{safe}</span>'


def render_chips(items: list[str], kind: str, max_items: int = 10) -> None:
    if not items:
        return
    shown = items[:max_items]
    html = " ".join(chip(x, kind) for x in shown)
    if len(items) > max_items:
        html += " " + chip(f"+{len(items) - max_items} meer", "neutral")
    st.markdown(html, unsafe_allow_html=True)


def bool_label(value: Any) -> str:
    if value is True or str(value).lower() == "true":
        return "✅ pass"
    if value is False or str(value).lower() == "false":
        return "❌ fail"
    return "—"


def parse_rule_reason(rule_reason: Any, selection: Any) -> pd.DataFrame:
    """Parse de rule_reason-string uit rules.py naar een compacte checktabel."""
    if rule_reason is None or pd.isna(rule_reason):
        return pd.DataFrame()

    text_value = str(rule_reason)
    side = str(selection or "").upper()
    if not side:
        return pd.DataFrame()

    # Pak alleen het HOME:/AWAY:-deel dat hoort bij de selectie.
    parts = [p.strip() for p in text_value.split(" | ")]
    side_part = next((p for p in parts if p.upper().startswith(f"{side}:")), text_value)
    side_part = side_part.replace(f"{side}:", "", 1).replace(f"{side.lower()}:", "", 1)

    rows = []
    label_map = {
        "prob": "Probability",
        "value": "Value",
        "rating_gap": "Rating gap",
        "odds": "Odds range",
        "snapshots": "Snapshots",
        "drift_pct": "Drift",
        "edge": "Home edge",
        "final": "Final rule",
    }

    for raw_check in [x.strip() for x in side_part.split(";") if x.strip()]:
        fields: dict[str, str] = {}
        metric_name = None

        for i, token in enumerate(raw_check.split("|")):
            token = token.strip()
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            k = k.strip()
            v = v.strip()
            if i == 0:
                metric_name = k
            fields[k] = v

        if not metric_name:
            continue

        if metric_name == "final":
            ok = fields.get("final")
            rows.append({"Onderdeel": label_map[metric_name], "Waarde": ok, "Grens": "", "Status": bool_label(ok)})
            continue

        if metric_name == "drift_pct":
            ok = fields.get("drift_ok") or fields.get("ok")
            grens = f"abs>={fields.get('min_abs', '—')} en support<={fields.get('oppose_max', '—')}"
            detail = fields.get("drift_fail_detail")
            rows.append({
                "Onderdeel": label_map[metric_name],
                "Waarde": fields.get(metric_name, "—"),
                "Grens": grens,
                "Status": bool_label(ok) + (f" · {detail}" if detail and detail != "None" else ""),
            })
            continue

        ok = fields.get("ok")
        grens = fields.get("min") or fields.get("range") or ""
        rows.append({
            "Onderdeel": label_map.get(metric_name, metric_name),
            "Waarde": fields.get(metric_name, "—"),
            "Grens": grens,
            "Status": bool_label(ok),
        })

    return pd.DataFrame(rows)


def render_manual_check(row: pd.Series) -> None:
    selection = row.get("selection")
    checks = parse_rule_reason(row.get("rule_reason"), selection)

    if checks.empty:
        st.info("Geen rule_reason beschikbaar om te parsen.")
    else:
        st.dataframe(checks, width="stretch", hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Passes v2", bool_label(row.get("passes_danger_combo_v2")))
    c2.metric("No longshots", bool_label(row.get("passes_danger_combo_v2_no_longshots")))
    c3.metric("Snapshots", metric_value(row.get("n_snapshots")))
    c4.metric("Drift gekozen", fmt_pct(row.get("selected_drift_pct")))


def render_pick_cards(df: pd.DataFrame, empty_text: str) -> None:
    if df.empty:
        st.info(empty_text)
        return

    # Filters boven de kaarten.
    filter_cols = st.columns([1, 1, 1, 2])
    tiers = sorted([x for x in df.get("pick_tier", pd.Series(dtype=str)).dropna().unique()]) if "pick_tier" in df.columns else []
    types = sorted([x for x in df.get("pick_type", pd.Series(dtype=str)).dropna().unique()]) if "pick_type" in df.columns else []
    comps = sorted([x for x in df.get("competition", pd.Series(dtype=str)).dropna().unique()]) if "competition" in df.columns else []

    with filter_cols[0]:
        tier_filter = st.multiselect("Tier", tiers, default=[])
    with filter_cols[1]:
        type_filter = st.multiselect("Type", types, default=[])
    with filter_cols[2]:
        only_safe = st.checkbox("Verberg X/danger", value=False)
    with filter_cols[3]:
        comp_filter = st.multiselect("Competitie", comps, default=[])

    work = df.copy()
    if tier_filter and "pick_tier" in work.columns:
        work = work[work["pick_tier"].isin(tier_filter)]
    if type_filter and "pick_type" in work.columns:
        work = work[work["pick_type"].isin(type_filter)]
    if comp_filter and "competition" in work.columns:
        work = work[work["competition"].isin(comp_filter)]
    if only_safe:
        work = work[(work.get("pick_tier") != "X") & (work.get("danger_tags").fillna("").astype(str).str.len() == 0)]

    if work.empty:
        st.info("Geen picks binnen deze filters.")
        return

    st.caption(f"{len(work)} picks — scan de kaart, open ‘Waarom?’ voor de volledige rule-check.")

    for _, r in work.iterrows():
        title = f"{r.get('home_team', '—')} - {r.get('away_team', '—')}"
        date = r.get("date", "—")
        competition = r.get("competition", "—")
        selection = r.get("selection", "—")
        pick_type = r.get("pick_type", "—")
        danger = split_tags(r.get("danger_tags"))
        sector = split_tags(r.get("sector_tags"))
        badge = tier_badge(r.get("pick_tier"), r.get("pick_stars"))
        kind = tier_class(r.get("pick_tier"), danger)

        with st.container(border=True):
            top_left, top_mid, top_right = st.columns([4, 2, 1.2])
            with top_left:
                st.markdown(f'<div class="bm-card-title">{title}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bm-muted">{date} · {competition}</div>', unsafe_allow_html=True)
            with top_mid:
                st.markdown(chip(badge, kind) + " " + chip(str(pick_type), "neutral"), unsafe_allow_html=True)
            with top_right:
                st.metric("Advies", selection)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Odds", fmt_num(r.get("selected_odds"), 2))
            c2.metric("ECI-kans", fmt_pct(r.get("selected_prob")))
            c3.metric("Value", fmt_num(r.get("selected_value"), 3))
            c4.metric("Strength", fmt_num(selected_strength(r), 2))
            c5.metric("Drift", fmt_pct(r.get("selected_drift_pct")))

            reason = r.get("classification_reason")
            if reason is not None and not pd.isna(reason) and str(reason).strip():
                st.markdown(f"**Waarom:** {reason}")

            if danger:
                render_chips(danger, "bad", max_items=8)
            elif sector:
                # Toon alleen de meest bruikbare tags op kaartniveau.
                visible = [x for x in sector if x.startswith(("market:", "passes:", "dynamic_strong:", "strength:", "prob:", "odds:", "value:"))]
                render_chips(visible[:8] or sector[:8], "good", max_items=8)
            else:
                st.markdown(chip("geen extra tags", "neutral"), unsafe_allow_html=True)

            with st.expander("Waarom is dit een pick? / handmatige check", expanded=False):
                render_manual_check(r)

                st.markdown("**Kernstats**")
                stats = {
                    "run_id": r.get("run_id"),
                    "match_id": r.get("match_id"),
                    "selection": r.get("selection"),
                    "pick_type": r.get("pick_type"),
                    "tier": r.get("pick_tier"),
                    "stars": r.get("pick_stars"),
                    "odds_home": r.get("odds_home"),
                    "odds_draw": r.get("odds_draw"),
                    "odds_away": r.get("odds_away"),
                    "prob_home": r.get("prob_home"),
                    "prob_draw": r.get("prob_draw"),
                    "prob_away": r.get("prob_away"),
                    "bet_home": r.get("bet_home"),
                    "bet_draw": r.get("bet_draw"),
                    "bet_away": r.get("bet_away"),
                    "rating_gap": r.get("rating_gap"),
                    "n_snapshots": r.get("n_snapshots"),
                    "home_drift_pct": r.get("home_drift_pct"),
                    "away_drift_pct": r.get("away_drift_pct"),
                    "rule_strength": r.get("rule_strength"),
                    "rule_strength_adj": r.get("rule_strength_adj"),
                    "strength_bucket": r.get("strength_bucket"),
                    "outcome": r.get("outcome"),
                    "score": r.get("score"),
                }
                st.dataframe(pd.DataFrame([stats]).pipe(prepare_display_df), width="stretch", hide_index=True)

                if r.get("rule_reason") is not None and not pd.isna(r.get("rule_reason")):
                    st.markdown("**Ruwe rule_reason**")
                    st.code(str(r.get("rule_reason")), language="text")
                if sector:
                    st.markdown("**Alle sector tags**")
                    render_chips(sector, "neutral", max_items=40)
                if danger:
                    st.markdown("**Alle danger tags**")
                    render_chips(danger, "bad", max_items=40)

def fail_explanation(row: pd.Series) -> str:
    reason = str(row.get("fail_reason", "")).lower()

    if reason == "value":
        margin = row.get("value_margin", row.get("single_fail_margin"))
        return f"Value mist de grens met {fmt_num(abs(margin), 4)}."

    if reason == "prob":
        margin = row.get("prob_margin", row.get("single_fail_margin"))
        return f"Probability zit {fmt_num(abs(margin), 4)} onder de minimumgrens."

    if reason == "odds":
        margin = row.get("odds_margin", row.get("single_fail_margin"))
        return f"Odds vallen buiten de toegestane range met marge {fmt_num(margin, 4)}."

    if reason == "drift":
        drift = row.get("selected_drift", row.get("drift_pct"))
        return f"Drift faalt: markt beweegt tegen of onvoldoende mee. Drift: {fmt_pct(drift)}."

    if reason == "snap":
        needed = row.get("snap_needed")
        return f"Te weinig snapshots. Nog nodig: {fmt_num(needed, 0)}."

    if reason == "rating":
        margin = row.get("rating_margin", row.get("single_fail_margin"))
        return f"Rating gap mist de grens met {fmt_num(abs(margin), 2)}."

    if reason == "edge":
        margin = row.get("edge_margin", row.get("single_fail_margin"))
        return f"Home edge faalt. Edge margin: {fmt_num(margin, 2)}."

    return "Geen specifieke uitleg beschikbaar."


def render_research_cards(df: pd.DataFrame, kind: str, empty_text: str) -> None:
    if df.empty:
        st.info(empty_text)
        return

    st.caption(f"{len(df)} kandidaten — nieuwste run per wedstrijd, gesorteerd op potentie.")

    for _, r in df.iterrows():
        title = f"{r.get('home_team', '—')} - {r.get('away_team', '—')}"
        reason = r.get("fail_reason", "—")
        side = r.get("side", "—")

        odds = r.get("selected_odds") if "selected_odds" in r else r.get("odds")
        prob = r.get("selected_prob") if "selected_prob" in r else r.get("probability")
        value = r.get("selected_value") if "selected_value" in r else r.get("value_score")
        drift = r.get("selected_drift") if "selected_drift" in r else r.get("drift_pct")
        strength = r.get("strength")

        if strength is None or pd.isna(strength):
            strength = r.get("single_fail_raw_strength")

        with st.container(border=True):
            st.markdown(f"### {title}")
            st.caption(
                f"{r.get('date', '—')} · {r.get('competition', '—')} · "
                f"run {r.get('run_id', '—')} · match {r.get('match_id', '—')}"
            )

            top = st.columns([1, 1, 1, 1, 1])
            top[0].metric("Side", side)
            top[1].metric("Fail", reason)
            top[2].metric("Odds", fmt_num(odds, 2))
            top[3].metric("Prob", fmt_pct(prob))
            top[4].metric("Value", fmt_num(value, 3))

            mid = st.columns([1, 1, 1, 1, 1])
            mid[0].metric("Strength", fmt_num(strength, 2))
            mid[1].metric("Margin", fmt_num(r.get("single_fail_margin"), 4))
            mid[2].metric("Snapshots", metric_value(r.get("n_snapshots")))
            mid[3].metric("Drift", fmt_pct(drift))
            mid[4].metric("Snap nodig", metric_value(r.get("snap_needed")))

            st.markdown(f"**Interpretatie:** {fail_explanation(r)}")

            margin_cols = [
                "prob_margin",
                "value_margin",
                "odds_margin",
                "drift_margin",
                "rating_margin",
                "edge_margin",
                "single_fail_margin",
            ]
            existing_margin_cols = [c for c in margin_cols if c in r.index and not pd.isna(r.get(c))]

            if existing_margin_cols:
                margin_df = pd.DataFrame(
                    [{"onderdeel": c, "marge": r.get(c)} for c in existing_margin_cols]
                )
                margin_df["marge"] = pd.to_numeric(margin_df["marge"], errors="coerce").round(4)

                with st.expander("Marge-details", expanded=False):
                    st.dataframe(margin_df, width="stretch", hide_index=True)

            with st.expander("Alle details", expanded=False):
                st.dataframe(pd.DataFrame([r]).pipe(prepare_display_df), width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# Expliciete queries gebaseerd op de engine-code
# -----------------------------------------------------------------------------

SQL_RECENT_RUNS = """
WITH pick_counts AS (
    SELECT
        run_id,
        COUNT(*) AS picks,
        COUNT(*) FILTER (WHERE outcome IS NULL) AS open_picks,
        COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS settled_picks,
        COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
        COUNT(*) FILTER (WHERE outcome = 'LOSS') AS losses
    FROM public.picks_evaluated
    GROUP BY run_id
), snapshot_counts AS (
    SELECT run_id, COUNT(*) AS model_matches
    FROM public.model_match_snapshots
    GROUP BY run_id
), near_miss_counts AS (
    SELECT run_id, COUNT(*) AS near_misses
    FROM public.picks_near_miss_candidates
    GROUP BY run_id
), single_fail_counts AS (
    SELECT run_id, COUNT(*) AS single_fails
    FROM public.picks_single_fail_candidates
    GROUP BY run_id
)
SELECT
    r.run_id,
    r.config ->> 'generated_at' AS generated_at,
    r.created_by,
    r.use_cutoff,
    r.source_note,
    COALESCE(sc.model_matches, 0) AS wedstrijden,
    COALESCE(pc.picks, 0) AS picks,
    COALESCE(pc.open_picks, 0) AS open_picks,
    COALESCE(pc.settled_picks, 0) AS settled_picks,
    COALESCE(pc.wins, 0) AS wins,
    COALESCE(pc.losses, 0) AS losses,
    COALESCE(nm.near_misses, 0) AS near_misses,
    COALESCE(sf.single_fails, 0) AS single_fails
FROM public.picks_run r
LEFT JOIN pick_counts pc ON pc.run_id = r.run_id
LEFT JOIN snapshot_counts sc ON sc.run_id = r.run_id
LEFT JOIN near_miss_counts nm ON nm.run_id = r.run_id
LEFT JOIN single_fail_counts sf ON sf.run_id = r.run_id
ORDER BY r.run_id DESC
LIMIT 15;
"""

SQL_RECENT_RUNS_COMPACT = """
WITH pick_counts AS (
    SELECT
        run_id,
        COUNT(*) AS picks,
        COUNT(*) FILTER (WHERE outcome IS NULL) AS open_picks,
        COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS settled_picks,
        COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
        COUNT(*) FILTER (WHERE outcome = 'LOSS') AS losses
    FROM public.picks_evaluated
    GROUP BY run_id
), snapshot_counts AS (
    SELECT run_id, COUNT(*) AS wedstrijden
    FROM public.model_match_snapshots
    GROUP BY run_id
), near_miss_counts AS (
    SELECT run_id, COUNT(*) AS near_misses
    FROM public.picks_near_miss_candidates
    GROUP BY run_id
), single_fail_counts AS (
    SELECT run_id, COUNT(*) AS single_fails
    FROM public.picks_single_fail_candidates
    GROUP BY run_id
)
SELECT
    r.run_id,
    r.config ->> 'generated_at' AS generated_at,
    COALESCE(sc.wedstrijden, 0) AS wedstrijden,
    COALESCE(pc.picks, 0) AS picks,
    COALESCE(pc.open_picks, 0) AS open_picks,
    COALESCE(pc.settled_picks, 0) AS settled_picks,
    COALESCE(pc.wins, 0) AS wins,
    COALESCE(pc.losses, 0) AS losses,
    COALESCE(nm.near_misses, 0) AS near_misses,
    COALESCE(sf.single_fails, 0) AS single_fails
FROM public.picks_run r
LEFT JOIN pick_counts pc ON pc.run_id = r.run_id
LEFT JOIN snapshot_counts sc ON sc.run_id = r.run_id
LEFT JOIN near_miss_counts nm ON nm.run_id = r.run_id
LEFT JOIN single_fail_counts sf ON sf.run_id = r.run_id
ORDER BY r.run_id DESC
LIMIT 10;
"""

SQL_PICK_BASE = """
SELECT
    run_id,
    match_id,
    NULLIF(date::text, '')::date AS date,
    competition,
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
    home_drift_pct,
    away_drift_pct,
    selection,
    pick_type,
    pick_tier,
    pick_stars,
    CASE
        WHEN selection = 'HOME' THEN odds_home
        WHEN selection = 'AWAY' THEN odds_away
        WHEN selection = 'DRAW' THEN odds_draw
    END AS selected_odds,
    CASE
        WHEN selection = 'HOME' THEN prob_home
        WHEN selection = 'AWAY' THEN prob_away
        WHEN selection = 'DRAW' THEN prob_draw
    END AS selected_prob,
    CASE
        WHEN selection = 'HOME' THEN bet_home
        WHEN selection = 'AWAY' THEN bet_away
        WHEN selection = 'DRAW' THEN bet_draw
    END AS selected_value,
    rule_strength,
    rule_strength_adj,
    strength_bucket,
    rating_gap,
    n_snapshots,
    CASE
        WHEN selection = 'HOME' THEN home_drift_pct
        WHEN selection = 'AWAY' THEN away_drift_pct
    END AS selected_drift_pct,
    rule_passed,
    rule_reason,
    outcome,
    result,
    score,
    classification_reason,
    sector_tags,
    danger_tags,
    passes_danger_combo_v2,
    passes_danger_combo_v2_no_longshots
FROM public.picks_evaluated
WHERE selection IS NOT NULL
"""

SQL_TODAY_TOMORROW = """
WITH latest_run AS (
    SELECT MAX(run_id) AS run_id
    FROM public.picks_run
),
candidates AS (
""" + SQL_PICK_BASE + """
  AND run_id = (SELECT run_id FROM latest_run)
  AND NULLIF(date::text, '')::date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '1 day'
)
SELECT *
FROM candidates
ORDER BY date ASC, pick_tier NULLS LAST, rule_strength_adj DESC NULLS LAST;
"""

SQL_OPEN_PICKS = """
WITH latest_run AS (
    SELECT MAX(run_id) AS run_id
    FROM public.picks_run
),
candidates AS (
""" + SQL_PICK_BASE + """
  AND run_id = (SELECT run_id FROM latest_run)
  AND outcome IS NULL
  AND NULLIF(date::text, '')::date >= CURRENT_DATE
)
SELECT *
FROM candidates
ORDER BY date ASC, pick_tier NULLS LAST, rule_strength_adj DESC NULLS LAST
LIMIT 100;
"""

SQL_LATEST_PICKS = """
WITH candidates AS (
""" + SQL_PICK_BASE + """
), latest_per_match AS (
    SELECT DISTINCT ON (match_id)
        *
    FROM candidates
    ORDER BY match_id, run_id DESC
)
SELECT *
FROM latest_per_match
ORDER BY run_id DESC, date DESC NULLS LAST, rule_strength_adj DESC NULLS LAST
LIMIT 50;
"""

SQL_PICK_SUMMARY = """
SELECT
    COUNT(*) AS total_picks,
    COUNT(*) FILTER (WHERE outcome IS NULL) AS open_picks,
    COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS settled_picks,
    COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
    COUNT(*) FILTER (WHERE outcome = 'LOSS') AS losses,
    ROUND(
        CASE
            WHEN COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) = 0 THEN NULL
            ELSE (COUNT(*) FILTER (WHERE outcome = 'WIN'))::numeric /
                 (COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')))::numeric
        END,
        4
    ) AS hitrate
FROM public.picks_evaluated;
"""

SQL_SETTLED_ROI = """
WITH settled AS (
    SELECT
        outcome,
        CASE
            WHEN selection = 'HOME' THEN odds_home
            WHEN selection = 'AWAY' THEN odds_away
            WHEN selection = 'DRAW' THEN odds_draw
        END AS selected_odds
    FROM public.picks_evaluated
    WHERE selection IS NOT NULL
      AND outcome IN ('WIN','LOSS')
)
SELECT
    COUNT(*) AS bets,
    ROUND(SUM(CASE WHEN outcome = 'WIN' THEN selected_odds - 1 ELSE -1 END)::numeric, 2) AS profit,
    ROUND((SUM(CASE WHEN outcome = 'WIN' THEN selected_odds - 1 ELSE -1 END) / NULLIF(COUNT(*), 0))::numeric, 4) AS roi
FROM settled;
"""

SQL_TIER_SUMMARY = """
WITH settled AS (
    SELECT
        COALESCE(NULLIF(pick_tier, ''), 'UNKNOWN') AS pick_tier,
        outcome,
        CASE
            WHEN selection = 'HOME' THEN odds_home
            WHEN selection = 'AWAY' THEN odds_away
            WHEN selection = 'DRAW' THEN odds_draw
        END AS selected_odds
    FROM public.picks_evaluated
    WHERE selection IS NOT NULL
      AND outcome IN ('WIN','LOSS')
)
SELECT
    pick_tier,
    COUNT(*) AS bets,
    COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
    ROUND((COUNT(*) FILTER (WHERE outcome = 'WIN'))::numeric / NULLIF(COUNT(*), 0), 4) AS hitrate,
    ROUND(SUM(CASE WHEN outcome = 'WIN' THEN selected_odds - 1 ELSE -1 END)::numeric, 2) AS profit,
    ROUND((SUM(CASE WHEN outcome = 'WIN' THEN selected_odds - 1 ELSE -1 END) / NULLIF(COUNT(*), 0))::numeric, 4) AS roi
FROM settled
GROUP BY pick_tier
ORDER BY
    CASE pick_tier
        WHEN 'A+' THEN 1 WHEN 'A' THEN 2 WHEN 'A-' THEN 3
        WHEN 'B' THEN 4 WHEN 'C' THEN 5 WHEN 'X' THEN 6 ELSE 7
    END;
"""

SQL_NEAR_MISS = """
SELECT
    run_id,
    NULLIF(date::text, '')::date AS date,
    competition,
    home_team,
    away_team,
    side,
    fail_reason,
    selected_odds,
    selected_prob,
    selected_value,
    selected_drift,
    strength,
    single_fail_margin,
    snap_needed,
    outcome,
    score
FROM public.picks_near_miss_candidates
WHERE side IS NOT NULL
ORDER BY run_id DESC, date ASC NULLS LAST, single_fail_margin DESC NULLS LAST
LIMIT 100;
"""

SQL_SINGLE_FAIL = """
SELECT
    run_id,
    NULLIF(date::text, '')::date AS date,
    competition,
    home_team,
    away_team,
    side,
    fail_reason,
    odds,
    probability,
    value_score,
    single_fail_raw_strength,
    single_fail_adj_strength,
    single_fail_calibrated_strength,
    single_fail_margin,
    snap_needed,
    n_snapshots,
    drift_pct,
    outcome,
    score
FROM public.picks_single_fail_candidates
WHERE side IS NOT NULL
ORDER BY run_id DESC, date ASC NULLS LAST, single_fail_margin DESC NULLS LAST
LIMIT 100;
"""

SQL_NEAR_MISS_OPEN_CARDS = """
WITH candidates AS (
    SELECT
        nm.run_id,
        nm.match_id,
        NULLIF(nm.date::text, '')::date AS date,
        nm.competition,
        nm.home_team,
        nm.away_team,
        nm.side,
        nm.fail_reason,

        nm.selected_odds,
        nm.selected_prob,
        nm.selected_value,
        nm.selected_drift,

        COALESCE(
            NULLIF(nm.strength, 0),
            CASE
                WHEN nm.side = 'HOME' THEN NULLIF(mms.raw_strength_home_all, 0)
                WHEN nm.side = 'AWAY' THEN NULLIF(mms.raw_strength_away_all, 0)
                ELSE NULL
            END,
            CASE
                WHEN nm.side = 'HOME' THEN NULLIF(mms.rule_strength_calibrated_home, 0)
                WHEN nm.side = 'AWAY' THEN NULLIF(mms.rule_strength_calibrated_away, 0)
                ELSE NULL
            END
        ) AS strength,

        nm.single_fail_margin,
        nm.prob_margin,
        nm.value_margin,
        nm.odds_margin,
        nm.drift_margin,
        nm.rating_margin,
        nm.edge_margin,
        nm.snap_needed,

        mms.n_snapshots,

        nm.outcome,
        nm.score
    FROM public.picks_near_miss_candidates nm
    LEFT JOIN public.model_match_snapshots mms
      ON mms.run_id = nm.run_id
     AND mms.match_id = nm.match_id
    WHERE nm.side IS NOT NULL
      AND nm.outcome IS NULL
      AND NULLIF(nm.date::text, '')::date >= CURRENT_DATE
),
latest_per_match AS (
    SELECT DISTINCT ON (match_id)
        *
    FROM candidates
    ORDER BY match_id, run_id DESC
)
SELECT *
FROM latest_per_match
ORDER BY strength DESC NULLS LAST, single_fail_margin DESC NULLS LAST
LIMIT 50;
"""

SQL_SINGLE_FAIL_OPEN_CARDS = """
WITH candidates AS (
    SELECT
        run_id,
        match_id,
        NULLIF(date::text, '')::date AS date,
        competition,
        home_team,
        away_team,
        side,
        fail_reason,

        odds,
        probability,
        value_score,

        single_fail_raw_strength,
        single_fail_adj_strength,
        single_fail_calibrated_strength,
        single_fail_margin,

        snap_needed,
        n_snapshots,
        drift_pct,

        outcome,
        score
    FROM public.picks_single_fail_candidates
    WHERE side IS NOT NULL
      AND outcome IS NULL
      AND NULLIF(date::text, '')::date >= CURRENT_DATE
),
latest_per_match AS (
    SELECT DISTINCT ON (match_id)
        *
    FROM candidates
    ORDER BY match_id, run_id DESC
)
SELECT *
FROM latest_per_match
ORDER BY single_fail_raw_strength DESC NULLS LAST, single_fail_margin DESC NULLS LAST
LIMIT 50;
"""

SQL_NEAR_MISS_SUMMARY = """
SELECT
    fail_reason,
    COUNT(*) AS candidates,
    COUNT(*) FILTER (WHERE outcome IS NULL) AS open,
    COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS settled,
    COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
    ROUND(
        CASE
            WHEN COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) = 0 THEN NULL
            ELSE COUNT(*) FILTER (WHERE outcome = 'WIN')::numeric
                 / COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS'))::numeric
        END,
        4
    ) AS hitrate,
    ROUND(AVG(selected_odds)::numeric, 3) AS avg_odds,
    ROUND(AVG(selected_prob)::numeric, 4) AS avg_prob,
    ROUND(AVG(selected_value)::numeric, 4) AS avg_value,
    ROUND(AVG(strength)::numeric, 3) AS avg_strength,
    ROUND(AVG(single_fail_margin)::numeric, 4) AS avg_margin
FROM public.picks_near_miss_candidates
WHERE side IS NOT NULL
GROUP BY fail_reason
ORDER BY candidates DESC;
"""

SQL_SINGLE_FAIL_SUMMARY = """
SELECT
    fail_reason,
    COUNT(*) AS candidates,
    COUNT(*) FILTER (WHERE outcome IS NULL) AS open,
    COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS settled,
    COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
    ROUND(
        CASE
            WHEN COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) = 0 THEN NULL
            ELSE COUNT(*) FILTER (WHERE outcome = 'WIN')::numeric
                 / COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS'))::numeric
        END,
        4
    ) AS hitrate,
    ROUND(AVG(odds)::numeric, 3) AS avg_odds,
    ROUND(AVG(probability)::numeric, 4) AS avg_prob,
    ROUND(AVG(value_score)::numeric, 4) AS avg_value,
    ROUND(AVG(single_fail_raw_strength)::numeric, 3) AS avg_raw_strength,
    ROUND(AVG(single_fail_calibrated_strength)::numeric, 3) AS avg_calibrated_strength,
    ROUND(AVG(single_fail_margin)::numeric, 4) AS avg_margin
FROM public.picks_single_fail_candidates
WHERE side IS NOT NULL
GROUP BY fail_reason
ORDER BY candidates DESC;
"""

SQL_COMPETITION_SUMMARY = """
WITH settled AS (
    SELECT
        competition,
        outcome,
        CASE
            WHEN selection = 'HOME' THEN odds_home
            WHEN selection = 'AWAY' THEN odds_away
            WHEN selection = 'DRAW' THEN odds_draw
        END AS selected_odds,
        rule_strength_adj
    FROM public.picks_evaluated
    WHERE selection IS NOT NULL
      AND outcome IN ('WIN','LOSS')
)
SELECT
    competition,
    COUNT(*) AS bets,
    COUNT(*) FILTER (WHERE outcome = 'WIN') AS wins,
    ROUND(COUNT(*) FILTER (WHERE outcome = 'WIN')::numeric / NULLIF(COUNT(*), 0), 4) AS hitrate,
    ROUND(SUM(CASE WHEN outcome = 'WIN' THEN selected_odds - 1 ELSE -1 END)::numeric, 2) AS profit,
    ROUND((SUM(CASE WHEN outcome = 'WIN' THEN selected_odds - 1 ELSE -1 END) / NULLIF(COUNT(*), 0))::numeric, 4) AS roi,
    ROUND(AVG(selected_odds)::numeric, 3) AS avg_odds,
    ROUND(AVG(rule_strength_adj)::numeric, 3) AS avg_strength
FROM settled
GROUP BY competition
HAVING COUNT(*) >= 3
ORDER BY roi DESC, bets DESC;
"""

SQL_ODDS_STATUS = """
SELECT
    MAX(captured_at) AS last_snapshot,
    COUNT(*) AS snapshots
FROM public.odds_values_snapshots;
"""

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("⚽ Betmobile Cockpit")
st.caption("Status, acties, picks, research en actieve regels — v0.10")

with st.sidebar:

    st.header("Dashboard")

    if st.button("Ververs dashboard", width="stretch"):
        st.cache_data.clear()
        st.success("Dashboard ververst")

    st.divider()

    st.header("Data")

    if st.button("Haal odds op", width="stretch"):

        with st.spinner("run_snapshot.py draait..."):

            code, stdout, stderr = run_python_script(
                SNAPSHOT_PATH,
                BASE_DIR,
            )

        if code == 0:
            st.success("Odds bijgewerkt")
            st.cache_data.clear()
        else:
            st.error("Snapshot mislukt")

    st.divider()

    st.header("Model")

    if st.button(
        "Odds ophalen + model draaien",
        type="primary",
        width="stretch"
    ):

        with st.spinner("Snapshot ophalen..."):

            code1, _, _ = run_python_script(
                SNAPSHOT_PATH,
                BASE_DIR,
            )

        if code1 == 0:

            with st.spinner("Model draaien..."):

                code2, _, _ = run_python_script(
                    RUN_MODEL_PATH,
                    ECI_ENGINE_DIR,
                )

            if code2 == 0:
                st.success("Snapshot + model voltooid")
                st.cache_data.clear()
            else:
                st.error("Model mislukt")

        else:
            st.error("Snapshot mislukt")

    if st.button(
        "Alleen model draaien",
        width="stretch"
    ):

        with st.spinner("run_model.py draait..."):

            code, stdout, stderr = run_python_script(
                RUN_MODEL_PATH,
                ECI_ENGINE_DIR,
            )

        if code == 0:
            st.success("Model succesvol afgerond")
            st.cache_data.clear()
        else:
            st.error("Model mislukt")

    st.divider()

    st.header("Onderhoud")

    if st.button(
        "Settle picks",
        width="stretch"
    ):

        with st.spinner("settle_picks.py draait..."):

            code, _, _ = run_python_script(
                SETTLE_PATH,
                BASE_DIR,
            )

        if code == 0:
            st.success("Settle voltooid")
            st.cache_data.clear()
        else:
            st.error("Settle mislukt")

    st.divider()

tab_status, tab_picks, tab_research, tab_rules, tab_debug = st.tabs(["Status", "Picks", "Research", "Regels", "Debug"])

with tab_status:
    st.subheader("Status")
    
    st.subheader("Systeemstatus")

    odds = query_df(SQL_ODDS_STATUS)

    odds_time = "—"
    if not odds.empty:
        odds_time = odds.iloc[0]["last_snapshot"]

    model_time = "—"
    runs_preview = query_df(SQL_RECENT_RUNS)

    if not runs_preview.empty:
        model_time = runs_preview.iloc[0]["generated_at"]

    today_picks = query_df(SQL_TODAY_TOMORROW)
    open_picks = query_df(SQL_OPEN_PICKS)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Odds snapshot",
        pd.to_datetime(odds_time).strftime("%H:%M"),
        age_text(odds_time),
    )

    c2.metric(
        "Model run",
        pd.to_datetime(model_time).strftime("%H:%M"),
        age_text(model_time),
    )

    c3.metric(
        "Open picks",
        len(open_picks),
    )

    c4.metric(
        "Vandaag+morgen",
        len(today_picks),
    )

    st.divider()

    try:
        runs = query_df(SQL_RECENT_RUNS)
        if runs.empty:
            st.info("Nog geen modelruns gevonden.")
        else:
            last = runs.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Laatste run", metric_value(last["run_id"]))
            c2.metric("Wedstrijden", metric_value(last["wedstrijden"]))
            c3.metric("Picks", metric_value(last["picks"]))
            c4.metric("Near misses", metric_value(last["near_misses"]))
            c5.metric("Single fails", metric_value(last["single_fails"]))

            st.write(f"**Generated at:** {last.get('generated_at', '—')}")
            with st.expander("Recente run-id's en aantallen", expanded=False):
                show_df(query_df(SQL_RECENT_RUNS_COMPACT), "Geen recente runs gevonden.")

        st.subheader("Historische pickstatus")
        summary = query_df(SQL_PICK_SUMMARY)
        roi = query_df(SQL_SETTLED_ROI)
        if not summary.empty:
            s = summary.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Totaal", metric_value(s["total_picks"]))
            c2.metric("Open", metric_value(s["open_picks"]))
            c3.metric("Settled", metric_value(s["settled_picks"]))
            c4.metric("Wins", metric_value(s["wins"]))
            c5.metric("Hitrate", metric_value(s["hitrate"]))
        if not roi.empty:
            r = roi.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Bets", metric_value(r["bets"]))
            c2.metric("Profit", metric_value(r["profit"]))
            c3.metric("ROI", metric_value(r["roi"]))

        st.subheader("Resultaat per tier")
        show_df(query_df(SQL_TIER_SUMMARY), "Nog geen gesettelde tier-data.")

        try:
            odds = query_df(SQL_ODDS_STATUS)
            if not odds.empty:
                st.subheader("Odds snapshots")
                o = odds.iloc[0]
                st.write(f"**Laatste snapshot:** {o['last_snapshot']}  ")
                st.write(f"**Aantal snapshots:** {o['snapshots']}")
        except Exception:
            st.info("Odds snapshot-status kon niet worden gelezen; niet blokkerend.")

    except Exception as exc:
        st.error("Status kon niet worden geladen.")
        st.exception(exc)


with tab_picks:
    st.subheader("Picks")

    view = st.radio(
        "Weergave",
        ["Vandaag + morgen", "Open picks", "Near misses", "Single fails", "Laatste 50"],
        horizontal=True,
    )

    try:
        if view == "Vandaag + morgen":
            df = query_df(SQL_TODAY_TOMORROW)
            render_pick_cards(df, "Geen picks voor vandaag of morgen.")
            with st.expander("Tabelweergave", expanded=False):
                show_df(df, "Geen picks voor vandaag of morgen.")
        elif view == "Open picks":
            df = query_df(SQL_OPEN_PICKS)
            render_pick_cards(df, "Geen open picks gevonden.")
            with st.expander("Tabelweergave", expanded=False):
                show_df(df, "Geen open picks gevonden.")
        elif view == "Laatste 50":
            df = query_df(SQL_LATEST_PICKS)
            render_pick_cards(df, "Geen picks gevonden.")
            with st.expander("Tabelweergave", expanded=False):
                show_df(df, "Geen picks gevonden.")
        elif view == "Near misses":
            df = query_df(SQL_NEAR_MISS)
            show_df(df, "Geen near misses gevonden.")
        else:
            df = query_df(SQL_SINGLE_FAIL)
            show_df(df, "Geen single fails gevonden.")
    except Exception as exc:
        st.error("Picks konden niet worden geladen.")
        st.exception(exc)

with tab_research:
    st.subheader("Research")

    research_view = st.radio(
        "Onderdeel",
        [
            "Near miss kansen",
            "Single fail kansen",
            "Near misses",
            "Single fails",
            "Competities",
            "Tier resultaat",
        ],
        horizontal=True,
    )

    try:
        if research_view == "Near miss kansen":
            st.markdown("### Open near misses met potentie")
            st.caption("Toekomstige near misses, gesorteerd op strength en marge.")
            df = query_df(SQL_NEAR_MISS_OPEN_CARDS)
            render_research_cards(df, "near_miss", "Geen open near misses gevonden.")

        elif research_view == "Single fail kansen":
            st.markdown("### Open single fails met potentie")
            st.caption("Toekomstige single fails, gesorteerd op calibrated strength en marge.")
            df = query_df(SQL_SINGLE_FAIL_OPEN_CARDS)
            render_research_cards(df, "single_fail", "Geen open single fails gevonden.")

        elif research_view == "Near misses":
            st.markdown("### Near misses per fail reason")
            st.caption("Laat zien welke bijna-picks vooral voorkomen en hoe ze historisch presteren.")
            df = query_df(SQL_NEAR_MISS_SUMMARY)
            show_df(df, "Geen near miss data gevonden.")

        elif research_view == "Single fails":
            st.markdown("### Single fails per fail reason")
            st.caption("Laat zien welke single-fail redenen interessant of juist gevaarlijk lijken.")
            df = query_df(SQL_SINGLE_FAIL_SUMMARY)
            show_df(df, "Geen single fail data gevonden.")

        elif research_view == "Competities":
            st.markdown("### Resultaat per competitie")
            st.caption("Alleen competities met minimaal 3 gesettelde picks.")
            df = query_df(SQL_COMPETITION_SUMMARY)
            show_df(df, "Geen competitie-data gevonden.")

        else:
            st.markdown("### Resultaat per tier")
            df = query_df(SQL_TIER_SUMMARY)
            show_df(df, "Nog geen tier-data gevonden.")

    except Exception as exc:
        st.error("Research-data kon niet worden geladen.")
        st.exception(exc)

with tab_rules:
    st.subheader("Actieve regels uit config.py")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min prob", ECI_RULE_PARAMS.get("min_prob"))
    c2.metric("Min value", ECI_RULE_PARAMS.get("min_value"))
    c3.metric("Odds range", f"{ECI_RULE_PARAMS.get('min_odds')}–{ECI_RULE_PARAMS.get('max_odds')}")
    c4.metric("Min snapshots", ECI_RULE_PARAMS.get("min_snapshots"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Min rating gap", ECI_RULE_PARAMS.get("min_rating_gap"))
    c2.metric("Min drift abs", ECI_RULE_PARAMS.get("min_drift_abs"))
    c3.metric("Min strength", MIN_STRENGTH)

    st.markdown("### Rule-engine")
    st.markdown(
        f"""
- HOME en AWAY gebruiken dezelfde basisgrenzen: probability, value, rating gap, odds range, snapshots en drift.
- HOME heeft daarnaast `rating_home_edge >= 0`.
- Gunstige drift: `<= {DRIFT_SUPPORT_THRESHOLD}`. Ongunstige drift: `>= {DRIFT_OPPOSE_THRESHOLD}`.
- Drift bonus/penalty: `+{DRIFT_SUPPORT_BONUS}` / `-{DRIFT_OPPOSE_PENALTY}`.
- Snapshot bonus: vanaf `{SNAP_BONUS_THRESHOLD}` snapshots `+{SNAP_BONUS}`.
- Range penalty: vanaf `{RANGE_PENALTY_THRESHOLD}` `-{RANGE_PENALTY}`.
        """
    )

    st.markdown("### Secondary picks")
    st.markdown(
        f"""
- Secondary picks actief: `{ENABLE_SECONDARY_PICKS}`.
- Toegestane fail: `{SECONDARY_ALLOWED_FAIL}`.
- Value tolerance: `{SECONDARY_VALUE_TOLERANCE}`.
- Min secondary strength: `{SECONDARY_MIN_STRENGTH}`.
- Min secondary probability: `{SECONDARY_MIN_PROB}`.
        """
    )

    st.markdown("### Ruwe config")
    st.json(
        {
            "ECI_RULE_PARAMS": ECI_RULE_PARAMS,
            "USE_CUTOFF_FEATURES": USE_CUTOFF_FEATURES,
            "MIN_STRENGTH": MIN_STRENGTH,
            "secondary": {
                "enabled": ENABLE_SECONDARY_PICKS,
                "allowed_fail": SECONDARY_ALLOWED_FAIL,
                "value_tolerance": SECONDARY_VALUE_TOLERANCE,
                "min_strength": SECONDARY_MIN_STRENGTH,
                "min_prob": SECONDARY_MIN_PROB,
            },
        }
    )


with tab_debug:
    st.subheader("Debug")
    st.code(
        f"""BASE_DIR={BASE_DIR}
ECI_ENGINE_DIR={ECI_ENGINE_DIR}
RUN_MODEL_PATH={RUN_MODEL_PATH}
RUN_MODEL_EXISTS={RUN_MODEL_PATH.exists()}
DB_DSN={DB_DSN}
""",
        language="text",
    )

    if st.button("Test databaseverbinding"):
        try:
            df = query_df("SELECT NOW() AS database_time;")
            st.success("Databaseverbinding werkt.")
            show_df(df, "Geen resultaat.")
        except Exception as exc:
            st.error("Databaseverbinding mislukt.")
            st.exception(exc)

    st.subheader("Kolommen per gebruikte tabel")
    for t in [
        "picks_run",
        "picks_evaluated",
        "model_match_snapshots",
        "picks_near_miss_candidates",
        "picks_single_fail_candidates",
        "odds_values_snapshots",
    ]:
        with st.expander(f"public.{t}"):
            show_df(table_columns(t), "Niet gevonden of geen kolommen.")
