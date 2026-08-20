"""
eci_quality.py

Odds-vrije kwaliteitscontrole van ECI zelf.

Andere vraag dan de stacking-test. Die vroeg: "weet ECI iets wat de markt niet
weet?" (antwoord: nee). Dit script vraagt: "klopt ECI op zichzelf, en waar
gaat het mis?" - zonder odds, zonder markt, zonder ROI.

Wat dit NIET kan opleveren: winst. Winst is ECI vs odds en die vraag is dicht.
Wat dit WEL oplevert: eerlijke labels (kloppen A/A+/B?), en concrete defecten
in het ratingmodel die aan de bron te repareren zijn.

Onderdelen:
1. KALIBRATIE      - claimt ECI wat hij levert? Per bucket, klasse, uitkomst.
2. MONOTONIE       - rangschikt ECI correct? (zekerder = vaker raak?)
3. RATING LAG      - Feyenoord/NEC-hypothese: is ECI te traag? Meet of teams
                     met recent sterk bewegende rating structureel mis worden
                     ingeschat, en of recente vorm restinformatie bevat.
4. TIER AUDIT      - doen de vastgezette tiers wat ze beloven?

Gebruik:
    python eci_quality.py
    python eci_quality.py --export-csv --no-refresh
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from config import OUTPUT_DIR
from db import db_engine, refresh_source_views, relation_exists
from fit_calibration import (
    DEFAULT_SCHEMA,
    DEFAULT_SOURCE,
    load_match_frame,
    prepare_match_frame,
    print_header,
    print_table,
)

EXPORT_DIR = OUTPUT_DIR / "research"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PROB_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
GAP_BINS = [0, 100, 250, 500, 1000, 100000]
GAP_LABELS = ["0-100", "100-250", "250-500", "500-1000", "1000+"]
MIN_CELL = 30  # onder dit aantal is een cel niet te interpreteren


# =====================================================================
# BASIS
# =====================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Betrouwbaarheidsinterval voor een hitrate; eerlijker bij kleine n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (centre - half, centre + half)


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Per wedstrijd drie observaties: thuis, gelijk, uit."""
    frames = []
    for i, side in enumerate(["home", "draw", "away"]):
        part = pd.DataFrame(
            {
                "match_id": df["match_id"].to_numpy() if "match_id" in df.columns else df.index,
                "date_dt": df["date_dt"].to_numpy(),
                "competition": df["competition"].to_numpy(),
                "calibration_class": df["calibration_class"].to_numpy(),
                "outcome_type": side,
                "p_eci": df[f"mdl_{side}"].to_numpy(float),
                "hit": (df["y_idx"].to_numpy(int) == i).astype(float),
            }
        )
        if "rating_gap" in df.columns:
            part["rating_gap"] = df["rating_gap"].to_numpy()
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def calibration_table(long: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Geclaimde kans vs werkelijke hitrate, met Wilson-interval."""
    df = long.copy()
    df["prob_bucket"] = pd.cut(df["p_eci"], bins=PROB_BINS, include_lowest=True)
    keys = (by or []) + ["prob_bucket"]

    out = (
        df.groupby(keys, observed=True)
        .agg(n=("hit", "size"), wins=("hit", "sum"), claimed=("p_eci", "mean"), actual=("hit", "mean"))
        .reset_index()
    )
    out["gap"] = out["actual"] - out["claimed"]
    ci = [wilson_ci(int(w), int(n)) for w, n in zip(out["wins"], out["n"])]
    out["ci_low"] = [c[0] for c in ci]
    out["ci_high"] = [c[1] for c in ci]
    # Significant scheef = de geclaimde kans valt buiten het interval.
    out["verdict"] = np.where(
        out["n"] < MIN_CELL, "te weinig data",
        np.where(out["claimed"] > out["ci_high"], "TE ZELFVERZEKERD",
                 np.where(out["claimed"] < out["ci_low"], "te voorzichtig", "ok")),
    )
    out["prob_bucket"] = out["prob_bucket"].astype(str)
    return out


# =====================================================================
# RATING LAG / VORM
# =====================================================================

def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Per team per wedstrijd: rating, ratingverandering, recente puntenoogst.

    Alles strikt op basis van WEDSTRIJDEN DAARVOOR (shift), zodat er nooit
    informatie uit de toekomst in een feature lekt.
    """
    if not {"home_team", "away_team", "home_rating", "away_rating"} <= set(df.columns):
        return pd.DataFrame()

    rows = []
    for side, team_col, rating_col in [
        ("home", "home_team", "home_rating"),
        ("away", "away_team", "away_rating"),
    ]:
        part = pd.DataFrame(
            {
                "match_key": df.index,
                "date_dt": df["date_dt"].to_numpy(),
                "team": df[team_col].to_numpy(),
                "rating": pd.to_numeric(df[rating_col], errors="coerce").to_numpy(),
                "side": side,
            }
        )
        y = df["y_idx"].to_numpy(int)
        if side == "home":
            part["points"] = np.select([y == 0, y == 1], [3.0, 1.0], default=0.0)
        else:
            part["points"] = np.select([y == 2, y == 1], [3.0, 1.0], default=0.0)
        rows.append(part)

    hist = pd.concat(rows, ignore_index=True).sort_values(["team", "date_dt"])
    g = hist.groupby("team", sort=False)

    # Ratingbeweging over de laatste 5 wedstrijden (alleen verleden).
    hist["rating_prev5"] = g["rating"].shift(5)
    hist["rating_delta5"] = hist["rating"] - hist["rating_prev5"]

    # Vorm: punten uit de vorige 3 wedstrijden (huidige wedstrijd uitgesloten).
    hist["form3"] = g["points"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    hist["n_prior"] = g.cumcount()
    return hist


def rating_lag_analysis(df: pd.DataFrame, hist: pd.DataFrame):
    """
    Feyenoord/NEC-hypothese, in twee vormen.

    A) Beweegt de rating traag? -> als een team recent flink STEEG, is ECI dan
       nog te pessimistisch over dat team (en omgekeerd)?
    B) Bevat recente vorm restinformatie bovenop de ECI-kans?
    """
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    keep = ["match_key", "side", "rating_delta5", "form3", "n_prior"]
    h = hist[keep]
    home = h[h["side"] == "home"].set_index("match_key")
    away = h[h["side"] == "away"].set_index("match_key")

    base = df.copy()
    base["home_delta5"] = home["rating_delta5"].reindex(base.index)
    base["away_delta5"] = away["rating_delta5"].reindex(base.index)
    base["home_form3"] = home["form3"].reindex(base.index)
    base["away_form3"] = away["form3"].reindex(base.index)
    base["home_prior"] = home["n_prior"].reindex(base.index)
    base["away_prior"] = away["n_prior"].reindex(base.index)

    # Thuisperspectief: werkelijke uitkomst (1/0.5/0) vs ECI-verwachting.
    y = base["y_idx"].to_numpy(int)
    base["actual_home"] = np.select([y == 0, y == 1], [1.0, 0.5], default=0.0)
    base["expected_home"] = base["mdl_home"] + 0.5 * base["mdl_draw"]
    base["resid_home"] = base["actual_home"] - base["expected_home"]

    base["delta_diff"] = base["home_delta5"] - base["away_delta5"]
    base["form_diff"] = base["home_form3"] - base["away_form3"]

    # --- A) ratingmomentum ---
    a = base.dropna(subset=["delta_diff", "resid_home"]).copy()
    a["momentum_bucket"] = pd.cut(
        a["delta_diff"],
        bins=[-np.inf, -100, -40, -10, 10, 40, 100, np.inf],
        labels=["thuis <<-100", "-100..-40", "-40..-10", "-10..+10", "+10..+40", "+40..+100", "thuis >>+100"],
    )
    tbl_a = (
        a.groupby("momentum_bucket", observed=True)
        .agg(n=("resid_home", "size"), avg_expected=("expected_home", "mean"),
             avg_actual=("actual_home", "mean"), resid=("resid_home", "mean"))
        .reset_index()
    )
    tbl_a["se"] = a.groupby("momentum_bucket", observed=True)["resid_home"].std().to_numpy() / np.sqrt(tbl_a["n"])
    tbl_a["t"] = tbl_a["resid"] / tbl_a["se"]

    # --- B) vorm als restinformatie ---
    b = base.dropna(subset=["form_diff", "resid_home"])
    b = b[(b["home_prior"] >= 5) & (b["away_prior"] >= 5)].copy()
    b["form_bucket"] = pd.cut(
        b["form_diff"],
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
        labels=["thuis veel slechter", "slechter", "gelijk", "beter", "thuis veel beter"],
    )
    tbl_b = (
        b.groupby("form_bucket", observed=True)
        .agg(n=("resid_home", "size"), avg_expected=("expected_home", "mean"),
             avg_actual=("actual_home", "mean"), resid=("resid_home", "mean"))
        .reset_index()
    )
    tbl_b["se"] = b.groupby("form_bucket", observed=True)["resid_home"].std().to_numpy() / np.sqrt(tbl_b["n"])
    tbl_b["t"] = tbl_b["resid"] / tbl_b["se"]

    # Slopes: resid ~ beta * feature. beta != 0 = restinformatie.
    stats = {}
    for name, part, col in [("rating_momentum", a, "delta_diff"), ("form3", b, "form_diff")]:
        if len(part) < 100:
            continue
        x = part[col].to_numpy(float)
        x = x - x.mean()
        r = part["resid_home"].to_numpy(float)
        beta = float(np.sum(x * r) / np.sum(x * x))
        resid = r - beta * x
        se = float(np.sqrt(np.sum(resid**2) / (len(x) - 2) / np.sum(x * x)))
        stats[name] = {"beta": beta, "se": se, "t": beta / se if se else np.nan, "n": len(part)}

    return tbl_a, tbl_b, stats


# =====================================================================
# TIER AUDIT
# =====================================================================

def tier_audit() -> pd.DataFrame:
    """Doen de vastgezette tiers wat ze beloven? Odds-vrij: claim vs realiteit."""
    source = "picks_evaluated_unique_v"
    exists, _ = relation_exists(source)
    if not exists:
        source = "picks_evaluated"
    q = f"""
        SELECT pick_tier, selection, outcome, prob_home, prob_draw, prob_away
        FROM public.{source}
        WHERE outcome IN ('WIN','LOSS') AND pick_tier IS NOT NULL
    """
    with db_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return df

    df["claimed"] = np.select(
        [df["selection"] == "HOME", df["selection"] == "DRAW", df["selection"] == "AWAY"],
        [df["prob_home"], df["prob_draw"], df["prob_away"]],
        default=np.nan,
    )
    df["claimed"] = pd.to_numeric(df["claimed"], errors="coerce")
    df.loc[df["claimed"] > 1.2, "claimed"] /= 100.0
    df["hit"] = (df["outcome"] == "WIN").astype(float)
    df = df.dropna(subset=["claimed"])

    out = (
        df.groupby("pick_tier", observed=True)
        .agg(n=("hit", "size"), wins=("hit", "sum"), claimed=("claimed", "mean"), actual=("hit", "mean"))
        .reset_index()
    )
    out["gap"] = out["actual"] - out["claimed"]
    ci = [wilson_ci(int(w), int(n)) for w, n in zip(out["wins"], out["n"])]
    out["ci_low"] = [c[0] for c in ci]
    out["ci_high"] = [c[1] for c in ci]
    return out.sort_values("claimed", ascending=False)


# =====================================================================
# HOOFDPROGRAMMA
# =====================================================================

def run(source: str, schema: str, refresh: bool, export_csv: bool,
        df: pd.DataFrame | None = None) -> dict:
    print_header("ECI KWALITEITSCONTROLE (ODDS-VRIJ)")
    print(
        "Vraag: klopt ECI op zichzelf, en waar gaat het mis?\n"
        "Dit meet GEEN winstgevendheid - die vraag is beantwoord met de odds erbij."
    )

    if df is None:
        if refresh:
            try:
                refresh_source_views()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] refresh mislukt, ga door: {exc}")
        df = prepare_match_frame(load_match_frame(source, schema))
    else:
        df = prepare_match_frame(df)
    df = df.reset_index(drop=True)

    if {"home_rating", "away_rating"} <= set(df.columns):
        hr = pd.to_numeric(df["home_rating"], errors="coerce")
        ar = pd.to_numeric(df["away_rating"], errors="coerce")
        df["rating_gap"] = (hr - ar).abs()

    long = to_long(df)

    # ---------- 1. KALIBRATIE ----------
    overall = calibration_table(long)
    print_table("1a. KALIBRATIE TOTAAL (claimed vs actual)", overall)

    by_outcome = calibration_table(long, by=["outcome_type"])
    print_table(
        "1b. KALIBRATIE PER UITKOMSTTYPE (gelijkspel is bij Elo vaak de zwakke plek)",
        by_outcome[by_outcome["n"] >= MIN_CELL],
        max_rows=80,
    )

    by_class = calibration_table(long, by=["calibration_class"])
    print_table("1c. KALIBRATIE PER COMPETITIEKLASSE", by_class[by_class["n"] >= MIN_CELL], max_rows=80)

    if "rating_gap" in df.columns:
        # Per wedstrijd de FAVORIET volgens ECI: claimt hij wat hij levert?
        # (Alle drie de uitkomsten samen middelen per definitie naar 1/3.)
        fav = df.dropna(subset=["rating_gap"]).copy()
        mat = fav[["mdl_home", "mdl_draw", "mdl_away"]].to_numpy(float)
        fav_idx = mat.argmax(axis=1)
        fav["fav_claimed"] = mat[np.arange(len(fav)), fav_idx]
        fav["fav_hit"] = (fav_idx == fav["y_idx"].to_numpy(int)).astype(float)
        fav["gap_bucket"] = pd.cut(fav["rating_gap"], bins=GAP_BINS, labels=GAP_LABELS)
        by_gap = (
            fav.groupby("gap_bucket", observed=True)
            .agg(n=("fav_hit", "size"), wins=("fav_hit", "sum"),
                 claimed=("fav_claimed", "mean"), actual=("fav_hit", "mean"))
            .reset_index()
        )
        by_gap["gap"] = by_gap["actual"] - by_gap["claimed"]
        ci = [wilson_ci(int(w), int(n)) for w, n in zip(by_gap["wins"], by_gap["n"])]
        by_gap["ci_low"] = [c[0] for c in ci]
        by_gap["ci_high"] = [c[1] for c in ci]
        by_gap["verdict"] = np.where(
            by_gap["n"] < MIN_CELL, "te weinig data",
            np.where(by_gap["claimed"] > by_gap["ci_high"], "TE ZELFVERZEKERD",
                     np.where(by_gap["claimed"] < by_gap["ci_low"], "te voorzichtig", "ok")),
        )
        print_table(
            "1d. KALIBRATIE VAN DE ECI-FAVORIET PER RATINGVERSCHIL",
            by_gap,
        )
    else:
        by_gap = pd.DataFrame()

    # ---------- 2. MONOTONIE ----------
    mono = overall[overall["n"] >= MIN_CELL][["prob_bucket", "n", "claimed", "actual"]].copy()
    inc = mono["actual"].diff().dropna()
    share_up = float((inc > 0).mean()) if len(inc) else np.nan
    corr = float(np.corrcoef(long["p_eci"], long["hit"])[0, 1])
    print_header("2. MONOTONIE (rangschikt ECI correct?)")
    print(mono.to_string(index=False))
    print(
        f"\nopeenvolgende buckets die stijgen: {share_up:.0%}\n"
        f"correlatie kans <-> uitkomst: {corr:.4f}\n"
        "Stijgend = ECI rangschikt correct, ook als de niveaus scheef staan.\n"
        "Rangschikken is repareerbaar met kalibratie; NIET rangschikken niet."
    )

    # ---------- 3. RATING LAG / VORM ----------
    hist = build_team_history(df)
    tbl_a, tbl_b, stats = rating_lag_analysis(df, hist)

    print_header("3. RATING LAG EN VORM (de Feyenoord/NEC-vraag)")
    print(
        "resid = werkelijke thuisuitkomst (1/0.5/0) minus ECI-verwachting.\n"
        "resid > 0 betekent: ECI onderschat de thuisploeg in dat hokje."
    )
    if not tbl_a.empty:
        print_table("3a. NAAR RATINGMOMENTUM (verschil in ratingbeweging laatste 5)", tbl_a)
    if not tbl_b.empty:
        print_table("3b. NAAR RECENTE VORM (punten laatste 3, thuis vs uit)", tbl_b)

    for name, s in stats.items():
        print(
            f"[slope] {name}: beta={s['beta']:+.6f} (SE {s['se']:.6f}, t={s['t']:+.2f}, n={s['n']})"
        )
    print(
        "|t| < 2 = geen aantoonbare restinformatie; de trage rating kost dan\n"
        "gemiddeld niets. |t| > 2 met consistente richting = wel een defect."
    )

    # ---------- 4. TIER AUDIT ----------
    try:
        tiers = tier_audit()
        print_header("4. TIER AUDIT (odds-vrij: belooft de tier wat hij levert?)")
        if tiers.empty:
            print("Geen picks met tier gevonden.")
        else:
            print(tiers.to_string(index=False))
            print(
                "\nLees zo: claimed hoort binnen [ci_low, ci_high] te vallen, en\n"
                "hogere tiers horen een hogere actual te hebben dan lagere."
            )
    except Exception as exc:  # noqa: BLE001
        tiers = pd.DataFrame()
        print(f"[warn] tier audit overgeslagen: {exc}")

    if export_csv:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, table in [
            ("eci_quality_calibration", overall),
            ("eci_quality_by_outcome", by_outcome),
            ("eci_quality_by_class", by_class),
            ("eci_quality_by_rating_gap", by_gap),
            ("eci_quality_rating_momentum", tbl_a),
            ("eci_quality_form", tbl_b),
            ("eci_quality_tiers", tiers),
        ]:
            if table is not None and not table.empty:
                path = EXPORT_DIR / f"{name}_{stamp}.csv"
                table.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"[export] {path}")

    return {"calibration": overall, "monotonic_share": share_up, "lag_stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Odds-vrije ECI-kwaliteitscontrole")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    args = parser.parse_args()
    run(
        source=args.source,
        schema=args.schema,
        refresh=not args.no_refresh,
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    main()