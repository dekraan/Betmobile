"""
research_momentum_historic.py
=============================

PRE-REGISTRATIE
---------------
Hypothese H-MOM: het ratingmomentum van een team voorspelt de
wedstrijduitkomst bovenop de kans die ECI zelf publiceert. Als dat zo is,
verwerkt ECI zijn eigen ratingveranderingen onvoldoende in zijn kansen.

  Horizon:      28 dagen. Eén horizon, vooraf vastgelegd. Eerder bleken
                vier gemeten horizonnen 0,60-0,89 met elkaar te
                correleren; meerdere toetsen zou zoeken zijn, geen test.
  Variabele:    momentum_thuis - momentum_uit, gestandaardiseerd.
                momentum = rating vlak voor de wedstrijd
                           minus rating 28 dagen daarvoor.
  Baseline:     de door ECI gepubliceerde kansen (home/draw/away).
  Model:        ordered probit met de ECI-kans als offset;
                latent = z_eci + beta * momentum_diff.
  Nulhypothese: beta = 0.
  Toets:        tweezijdig, alpha = 0.05, cluster-robuuste standaardfout
                geclusterd op competitie, kritieke waarde t(k-1).

Dit is een retrospectieve toets op historische data en valt buiten de
out-of-sample freeze op het rule engine. Een positieve uitkomst is
in-sample hypothesegeneratie en GEEN bewijs van edge; die vraagt een
prospectieve toets.

Ratings komen strikt van VOOR de wedstrijddatum, dus geen lookahead.

Gebruik
-------
    python research_momentum_historic.py
    python research_momentum_historic.py --horizon 28
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm, t as tdist
    from scipy.optimize import minimize_scalar
except ImportError:
    sys.exit("scipy ontbreekt: pip install scipy")

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 ontbreekt: pip install psycopg2-binary")

try:
    from betmobile_settings import DB_CONFIG
except ImportError:
    sys.exit("betmobile_settings.py niet gevonden. Draai vanuit je Betmobile-map.")


HERE = Path(__file__).resolve().parent
RATINGS = HERE / "eci_history" / "ratings_history.jsonl"

HORIZON_DAYS = 28          # pre-registered
TAU_FALLBACK = 0.3467
EPS = 1e-9


def hr(t): print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def norm_name(s) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(s.lower().split())


# ---------------------------------------------------------------------------

def load_ratings():
    if not RATINGS.exists():
        sys.exit(f"Niet gevonden: {RATINGS}")
    print("  ratings inlezen ...")
    df = pd.read_json(RATINGS, lines=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["eci_team_id", "date"])

    series = {str(tid): (sub["date"].values, sub["points"].values)
              for tid, sub in df.groupby("eci_team_id")}

    names = {}
    for tid, name in df[["eci_team_id", "team_name"]].drop_duplicates().values:
        k = norm_name(name)
        if k:
            names.setdefault(k, str(tid))

    print(f"  {len(df):,} punten, {len(series):,} teams")
    return series, names


def rating_before(series, tid, when):
    """Laatste weekpunt strikt voor 'when'."""
    e = series.get(str(tid))
    if e is None:
        return None
    dates, pts = e
    i = np.searchsorted(dates, np.datetime64(when), side="left")
    return float(pts[i - 1]) if i > 0 else None


def load_matches() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_session(readonly=True, autocommit=True)
    df = pd.read_sql("""
        SELECT date, home_team, away_team, competition, eci_score,
               home_win_pct, draw_pct, away_win_pct
        FROM eci_data
        WHERE eci_score IS NOT NULL AND eci_score <> ''
          AND home_win_pct IS NOT NULL
          AND COALESCE(is_excluded, false) = false
    """, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["home_win_pct"].max() > 1.5:
        for c in ("home_win_pct", "draw_pct", "away_win_pct"):
            df[c] = df[c] / 100.0

    # eci_score staat als "0-1"
    parts = df["eci_score"].astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    df["hg"] = pd.to_numeric(parts[0], errors="coerce")
    df["ag"] = pd.to_numeric(parts[1], errors="coerce")
    df = df.dropna(subset=["date", "hg", "ag"])

    df["outcome"] = np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2))
    return df.reset_index(drop=True)


def tau_per_competition(df) -> dict:
    """Drempel per competitie, uit de gepubliceerde gelijkspelkansen.

    Bij een ordered probit is P(gelijk) het grootst rond z=0, en dan
    geldt P(gelijk) = 2*Phi(tau) - 1. We schatten tau uit de wedstrijden
    met de kleinste onbalans tussen thuis en uit.
    """
    out = {}
    for comp, sub in df.groupby("competition"):
        near = sub[(sub.home_win_pct - sub.away_win_pct).abs() < 0.10]
        if len(near) < 20:
            continue
        pd_mean = near.draw_pct.mean()
        out[comp] = norm.ppf((pd_mean + 1) / 2)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def latent_from_probs(p_home, p_away, tau):
    """Keer de gepubliceerde kansen om naar de latente variabele z."""
    p_home = np.clip(p_home, 1e-6, 1 - 1e-6)
    p_away = np.clip(p_away, 1e-6, 1 - 1e-6)
    z1 = tau - norm.ppf(1 - p_home)
    z2 = -tau - norm.ppf(p_away)
    return (z1 + z2) / 2.0


def loglik_per_obs(beta, z0, x, tau, y):
    z = z0 + beta * x
    p_home = 1 - norm.cdf(tau - z)
    p_away = norm.cdf(-tau - z)
    p_draw = np.clip(1 - p_home - p_away, EPS, 1)
    p = np.where(y == 0, np.clip(p_home, EPS, 1),
                 np.where(y == 1, p_draw, np.clip(p_away, EPS, 1)))
    return np.log(p)


def fit_beta(z0, x, tau, y):
    f = lambda b: -loglik_per_obs(b, z0, x, tau, y).sum()
    res = minimize_scalar(f, bounds=(-3, 3), method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x)


def cluster_robust_se(beta, z0, x, tau, y, clusters, h=1e-4):
    """Sandwich-schatter, geclusterd op competitie."""
    ll_p = loglik_per_obs(beta + h, z0, x, tau, y)
    ll_m = loglik_per_obs(beta - h, z0, x, tau, y)
    ll_0 = loglik_per_obs(beta, z0, x, tau, y)

    score = (ll_p - ll_m) / (2 * h)
    hess = ((ll_p - 2 * ll_0 + ll_m) / (h ** 2)).sum()
    if hess >= 0:
        return np.nan, 0

    meat = 0.0
    uniq = pd.unique(clusters)
    for c in uniq:
        meat += score[clusters == c].sum() ** 2

    bread = 1.0 / hess
    var = bread * meat * bread
    return float(np.sqrt(abs(var))), len(uniq)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="H-MOM: ratingmomentum bovenop ECI")
    ap.add_argument("--horizon", type=int, default=HORIZON_DAYS,
                    help=f"dagen (pre-registered: {HORIZON_DAYS})")
    ap.add_argument("--no-european", action="store_true",
                    help="Champions/Europa/Conference League weglaten")
    ap.add_argument("--cluster", choices=["competition", "team_season"],
                    default="competition",
                    help="cluster-eenheid voor de standaardfout")
    args = ap.parse_args()

    if args.horizon != HORIZON_DAYS:
        print(f"\n  LET OP: horizon {args.horizon} wijkt af van de "
              f"geregistreerde {HORIZON_DAYS} dagen.")
        print("  Meerdere horizonnen proberen is zoeken, geen toetsen.\n")

    if args.no_european or args.cluster != "competition":
        print("\n  ROBUUSTHEIDSVARIANT. De geregistreerde toets is:")
        print("  alle competities, geclusterd op competitie. Rapporteer dit")
        print("  als robuustheidscheck, niet als de toets zelf.\n")

    hr("H-MOM: voorspelt ratingmomentum iets bovenop de ECI-kans?")
    series, names = load_ratings()
    df = load_matches()
    print(f"  {len(df):,} afgespeelde wedstrijden met uitslag en kansen")
    print(f"  {df.date.min():%Y-%m-%d} tot {df.date.max():%Y-%m-%d}")

    if args.no_european:
        mask = df.competition.astype(str).str.contains(
            "Champions League|Europa League|Conference League",
            case=False, regex=True, na=False)
        print(f"  Europese toernooien weggelaten: {mask.sum():,} wedstrijden")
        df = df[~mask].reset_index(drop=True)

    df["hid"] = df.home_team.map(lambda n: names.get(norm_name(n)))
    df["aid"] = df.away_team.map(lambda n: names.get(norm_name(n)))
    df = df.dropna(subset=["hid", "aid"])
    print(f"  gekoppeld op naam: {len(df):,}")

    delta = pd.Timedelta(days=args.horizon)
    mom = []
    for r in df.itertuples():
        h_now = rating_before(series, r.hid, r.date)
        a_now = rating_before(series, r.aid, r.date)
        h_then = rating_before(series, r.hid, r.date - delta)
        a_then = rating_before(series, r.aid, r.date - delta)
        if None in (h_now, a_now, h_then, a_then):
            mom.append(np.nan)
        else:
            mom.append((h_now - h_then) - (a_now - a_then))

    df = df.assign(mom_diff=mom).dropna(subset=["mom_diff"]).reset_index(drop=True)
    print(f"  met momentum over {args.horizon} dagen: {len(df):,}")

    taus = tau_per_competition(df)
    df["tau"] = df.competition.map(lambda c: taus.get(c, TAU_FALLBACK))

    df["z0"] = latent_from_probs(df.home_win_pct.values,
                                 df.away_win_pct.values, df.tau.values)

    sd = df.mom_diff.std()
    df["x"] = df.mom_diff / sd
    print(f"\n  momentumverschil: sd = {sd:.1f} ratingpunten, "
          f"bereik {df.mom_diff.min():.0f} tot {df.mom_diff.max():.0f}")

    # --- beschrijvend -----------------------------------------------------
    hr("Beschrijvend: uitkomst per momentumkwintiel")
    df["q"] = pd.qcut(df.mom_diff, 5, labels=["laag", "2", "3", "4", "hoog"])
    print(f"  {'kwintiel':<10} {'n':>6} {'gem mom':>9} "
          f"{'ECI thuis':>10} {'echt thuis':>11} {'verschil':>9}")
    for q, sub in df.groupby("q", observed=True):
        pred = sub.home_win_pct.mean()
        act = (sub.outcome == 0).mean()
        print(f"  {str(q):<10} {len(sub):>6,} {sub.mom_diff.mean():>9.1f} "
              f"{pred*100:>9.1f}% {act*100:>10.1f}% {(act-pred)*100:>8.1f}pp")

    # --- toets ------------------------------------------------------------
    hr("Toets")
    z0 = df.z0.values
    x = df.x.values
    tau = df.tau.values
    y = df.outcome.values

    ll0 = loglik_per_obs(0.0, z0, x, tau, y).sum()
    beta = fit_beta(z0, x, tau, y)
    ll1 = loglik_per_obs(beta, z0, x, tau, y).sum()

    if args.cluster == "team_season":
        # Seizoen loopt van juli tot juni; cluster op thuisploeg per seizoen.
        season = np.where(df.date.dt.month >= 7, df.date.dt.year,
                          df.date.dt.year - 1)
        clusters = df.hid.astype(str) + "_" + pd.Series(season, index=df.index).astype(str)
        clusters = clusters.values
    else:
        clusters = df.competition.values

    se, k = cluster_robust_se(beta, z0, x, tau, y, clusters)
    crit = tdist.ppf(0.975, k - 1) if k > 1 else np.nan
    tstat = beta / se if se and not np.isnan(se) else np.nan

    print(f"  n                     {len(df):,}")
    print(f"  cluster-eenheid       {args.cluster}")
    print(f"  aantal clusters       {k:>8,}")
    print(f"  beta (per sd)         {beta:+.4f}")
    print(f"  cluster-robuuste se   {se:.4f}")
    print(f"  t                     {tstat:+.2f}")
    print(f"  kritieke waarde       {crit:.3f}   (t met {k-1} vrijheidsgraden)")
    print(f"\n  log loss zonder momentum  {-ll0/len(df):.5f}")
    print(f"  log loss met momentum     {-ll1/len(df):.5f}")
    print(f"  verbetering               {(ll1-ll0)/len(df):.6f} per wedstrijd")

    hr("CONCLUSIE")
    if np.isnan(tstat):
        print("  Standaardfout niet te bepalen; controleer de invoer.")
    elif abs(tstat) > crit:
        richting = "positief" if beta > 0 else "negatief"
        print(f"  H0 verworpen (|t| = {abs(tstat):.2f} > {crit:.3f}).")
        print(f"  Momentum draagt {richting} bij bovenop de ECI-kans.")
        print()
        print("  Dit is in-sample hypothesegeneratie, geen aangetoonde edge.")
        print("  Een effect tegenover ECI is nog geen effect tegenover de markt:")
        print("  de bookmaker kent dezelfde uitslagen als ECI.")
        print("  Vervolgstap is een prospectieve toets, en daarna pas CLV.")
    else:
        print(f"  H0 niet verworpen (|t| = {abs(tstat):.2f} < {crit:.3f}).")
        print("  Ratingmomentum voegt niets toe bovenop de ECI-kans.")
        print()
        print("  Daarmee is de momentumhypothese verworpen op historische data,")
        print("  en vervalt de reden voor de prospectieve toets in oktober.")

    out = HERE / "momentum_historic.csv"
    df[["date", "competition", "home_team", "away_team", "mom_diff",
        "home_win_pct", "draw_pct", "away_win_pct", "outcome"]].to_csv(out, index=False)
    print(f"\n  [OK] data -> {out.name}")


if __name__ == "__main__":
    main()