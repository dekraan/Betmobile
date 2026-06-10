import pandas as pd

from config import RULE_MIN_PROB, RULE_MIN_VALUE

def analyze_failures(df: pd.DataFrame):
    print("\n=== FAIL ANALYSIS ===")

    rows = []

    for _, r in df.iterrows():
        # HOME kant
        if r.get("home_fail_count", 0) > 0:
            rows.append({
                "match_id": r["match_id"],
                "competition": r["competition"],
                "side": "HOME",
                "fails": r["home_fail_reasons"],
                "fail_count": r["home_fail_count"],
                "prob": r["Home Prob"],
                "value": r["bet_home"],
                "odds": r["odds_home"],
                "drift_pct": r.get("home_drift_pct"),
                "drift_fail_detail": r.get("home_drift_fail_detail"),
                "n_snapshots": r.get("n_snapshots"),
            })

        # AWAY kant
        if r.get("away_fail_count", 0) > 0:
            rows.append({
                "match_id": r["match_id"],
                "competition": r["competition"],
                "side": "AWAY",
                "fails": r["away_fail_reasons"],
                "fail_count": r["away_fail_count"],
                "prob": r["Away Prob"],
                "value": r["bet_away"],
                "odds": r["odds_away"],
                "drift_pct": r.get("away_drift_pct"),
                "drift_fail_detail": r.get("away_drift_fail_detail"),
                "n_snapshots": r.get("n_snapshots"),
            })

    fail_df = pd.DataFrame(rows)

    if fail_df.empty:
        print("Geen fails gevonden.")
        return

    print(f"Totaal fail cases (home+away apart geteld): {len(fail_df)}")

    # 1. Single fails
    single = fail_df[fail_df["fail_count"] == 1].copy()
    print("\n--- SINGLE FAILS ---")
    if single.empty:
        print("Geen single fails.")
    else:
        print(single["fails"].value_counts().to_string())

    # 2. Double fails
    double = fail_df[fail_df["fail_count"] == 2].copy()
    print("\n--- DOUBLE FAILS ---")
    if double.empty:
        print("Geen double fails.")
    else:
        print(double["fails"].value_counts().head(20).to_string())

    # 3. Alles waar drift in zit
    drift_cases = fail_df[fail_df["fails"].fillna("").str.contains("drift")].copy()
    print("\n--- DRIFT INVOLVED ---")
    print(f"Totaal drift-cases: {len(drift_cases)}")

    drift_only = drift_cases[drift_cases["fail_count"] == 1].copy()
    drift_multi = drift_cases[drift_cases["fail_count"] > 1].copy()

    print(f"Drift only: {len(drift_only)}")
    print(f"Drift + andere fails: {len(drift_multi)}")

    print("\n--- DRIFT COMBINATIES ---")
    if drift_multi.empty:
        print("Geen drift-combinaties.")
    else:
        print(drift_multi["fails"].value_counts().head(20).to_string())

    print("\n--- DRIFT FAIL DETAIL ---")
    if drift_cases.empty:
        print("Geen drift-fails.")
    else:
        print(drift_cases["drift_fail_detail"].value_counts(dropna=False).to_string())        

    # 4. Interessant: drift fail, maar prob+value wel goed
    interesting = drift_cases[
        (drift_cases["prob"] >= RULE_MIN_PROB) &
        (drift_cases["value"] >= RULE_MIN_VALUE)
    ].copy()

    print("\n--- DRIFT FAIL, MAAR PROB + VALUE WEL GOED ---")
    print(f"Aantal: {len(interesting)}")

    if not interesting.empty:
        cols = [
            "match_id", "competition", "side",
            "fails", "fail_count",
            "prob", "value", "odds",
            "drift_pct", "n_snapshots"
        ]
        print(interesting[cols].head(20).to_string(index=False))