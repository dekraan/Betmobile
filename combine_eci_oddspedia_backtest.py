# combine_odds.py
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine


CURRENT_USER = "dekraan"

DB_DSN = "postgresql+psycopg2://postgres:300500@localhost:5432/Betmobile"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "Betmobile",
    "user": "postgres",
    "password": "300500",
}

MAPPING_FILE = Path(__file__).with_name("team_mappings_eci_oddspedia.json")
TARGET_TABLE = "eci_oddspedia_matches"


def safe_float(x):
    if x is None:
        return None
    try:
        s = str(x).strip()
        if not s or s.lower() in {"n/a", "na", "nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def load_team_mapping():
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["mapping"]["oddspedia_to_eci"]


def ensure_target_table():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS public.{TARGET_TABLE} (
                    match_id text PRIMARY KEY,
                    eci_date date,
                    oddspedia_date date,
                    date_diff_days integer,

                    home_team text,
                    away_team text,
                    oddspedia_home_team text,
                    oddspedia_away_team text,

                    odds_home double precision,
                    odds_draw double precision,
                    odds_away double precision,

                    home_win_pct double precision,
                    draw_pct double precision,
                    away_win_pct double precision,

                    implied_home double precision,
                    implied_draw double precision,
                    implied_away double precision,

                    value_home double precision,
                    value_draw double precision,
                    value_away double precision,

                    home_rating double precision,
                    away_rating double precision,
                    rating_diff double precision,

                    competition text,
                    oddspedia_competition text,

                    score text,
                    result text,
                    status text,

                    created_at timestamptz,
                    updated_at timestamptz,
                    created_by text,
                    updated_by text
                );
            """)
            conn.commit()


def get_data_from_database():
    print("\nFetching data from database...")
    engine = create_engine(DB_DSN)

    oddspedia_query = """
        SELECT
            date,
            home_team,
            away_team,
            odds_home,
            odds_draw,
            odds_away,
            score,
            status,
            competition
        FROM public.oddspedia_unibet_backbone
        WHERE date IS NOT NULL
    """

    eci_query = """
        SELECT
            date,
            home_team,
            away_team,
            home_win_pct,
            draw_pct,
            away_win_pct,
            home_rating,
            away_rating,
            competition
        FROM public.eci_data
        WHERE date IS NOT NULL
    """

    with engine.begin() as conn:
        df_odd = pd.read_sql(oddspedia_query, conn)
        df_eci = pd.read_sql(eci_query, conn)

    print(f"Retrieved {len(df_odd)} rows from oddspedia_unibet_backbone")
    print(f"Retrieved {len(df_eci)} rows from eci_data")

    return df_odd, df_eci


def parse_result(score):
    if not score:
        return None

    try:
        h, a = map(int, str(score).split("-"))
        if h > a:
            return "HOME"
        if a > h:
            return "AWAY"
        return "DRAW"
    except Exception:
        return None


def combine_data(df_oddspedia, df_eci):
    print("\nCombining data...")

    odd_to_eci = load_team_mapping()
    now_utc = datetime.now(timezone.utc)

    df_odd = df_oddspedia.copy()
    df_odd["date"] = pd.to_datetime(df_odd["date"], errors="coerce")
    df_odd = df_odd.dropna(subset=["date", "home_team", "away_team"])

    df_odd["home_team_mapped"] = df_odd["home_team"].map(odd_to_eci).fillna(df_odd["home_team"])
    df_odd["away_team_mapped"] = df_odd["away_team"].map(odd_to_eci).fillna(df_odd["away_team"])

    df_e = df_eci.copy()
    df_e["date"] = pd.to_datetime(df_e["date"], errors="coerce")
    df_e = df_e.dropna(subset=["date", "home_team", "away_team"])

    df_e["home_rating_num"] = df_e["home_rating"].apply(safe_float)
    df_e["away_rating_num"] = df_e["away_rating"].apply(safe_float)

    df_e["date_min"] = df_e["date"] - pd.Timedelta(days=1)
    df_e["date_max"] = df_e["date"] + pd.Timedelta(days=1)

    matched = []
    date_diffs = []

    print("\nMatching games between datasets...")

    for _, o in df_odd.iterrows():
        odds_date = o["date"]

        candidates = df_e[
            (df_e["date_min"] <= odds_date)
            & (df_e["date_max"] >= odds_date)
            & (df_e["home_team"] == o["home_team_mapped"])
            & (df_e["away_team"] == o["away_team_mapped"])
        ]

        if candidates.empty:
            continue

        eci = candidates.iloc[0]
        date_diff = abs((eci["date"] - odds_date).days)
        date_diffs.append(date_diff)

        odds_home = safe_float(o["odds_home"])
        odds_draw = safe_float(o["odds_draw"])
        odds_away = safe_float(o["odds_away"])

        home_pct = safe_float(eci["home_win_pct"])
        draw_pct = safe_float(eci["draw_pct"])
        away_pct = safe_float(eci["away_win_pct"])

        implied_home = (1 / odds_home) if odds_home else None
        implied_draw = (1 / odds_draw) if odds_draw else None
        implied_away = (1 / odds_away) if odds_away else None

        value_home = (home_pct * odds_home) if home_pct and odds_home else None
        value_draw = (draw_pct * odds_draw) if draw_pct and odds_draw else None
        value_away = (away_pct * odds_away) if away_pct and odds_away else None

        home_rating = eci["home_rating_num"]
        away_rating = eci["away_rating_num"]

        rating_diff = None
        if home_rating is not None and away_rating is not None:
            rating_diff = home_rating - away_rating

        eci_date_str = eci["date"].strftime("%Y-%m-%d")
        match_id = f"{eci_date_str}_{eci['home_team']}_{eci['away_team']}"

        matched.append({
            "match_id": match_id,
            "eci_date": eci["date"].date(),
            "oddspedia_date": odds_date.date(),
            "date_diff_days": int(date_diff),

            "home_team": eci["home_team"],
            "away_team": eci["away_team"],
            "oddspedia_home_team": o["home_team"],
            "oddspedia_away_team": o["away_team"],

            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,

            "home_win_pct": home_pct,
            "draw_pct": draw_pct,
            "away_win_pct": away_pct,

            "implied_home": implied_home,
            "implied_draw": implied_draw,
            "implied_away": implied_away,

            "value_home": value_home,
            "value_draw": value_draw,
            "value_away": value_away,

            "home_rating": home_rating,
            "away_rating": away_rating,
            "rating_diff": rating_diff,

            "competition": eci["competition"],
            "oddspedia_competition": o["competition"],

            "score": o["score"],
            "result": parse_result(o["score"]),
            "status": o["status"],

            "created_at": now_utc,
            "updated_at": now_utc,
            "created_by": CURRENT_USER,
            "updated_by": CURRENT_USER,
        })

    combined = pd.DataFrame(matched)

    print(f"\nMatched {len(combined)} out of {len(df_odd)} Oddspedia games")
    if date_diffs:
        print(f"Average date difference: {sum(date_diffs) / len(date_diffs):.2f}")
        print(f"Maximum date difference: {max(date_diffs)}")

    return combined


def save_to_database(combined_data):
    print(f"\nSaving to public.{TARGET_TABLE}...")

    if combined_data.empty:
        print("No matched data to save.")
        return

    combined_data = combined_data.replace({np.nan: None})

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE public.{TARGET_TABLE};")
            conn.commit()

    insert_sql = f"""
        INSERT INTO public.{TARGET_TABLE} (
            match_id,
            eci_date,
            oddspedia_date,
            date_diff_days,
            home_team,
            away_team,
            oddspedia_home_team,
            oddspedia_away_team,
            odds_home,
            odds_draw,
            odds_away,
            home_win_pct,
            draw_pct,
            away_win_pct,
            implied_home,
            implied_draw,
            implied_away,
            value_home,
            value_draw,
            value_away,
            home_rating,
            away_rating,
            rating_diff,
            competition,
            oddspedia_competition,
            score,
            result,
            status,
            created_at,
            updated_at,
            created_by,
            updated_by
        ) VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s
        )
        ON CONFLICT (match_id) DO UPDATE SET
            odds_home = EXCLUDED.odds_home,
            odds_draw = EXCLUDED.odds_draw,
            odds_away = EXCLUDED.odds_away,
            home_win_pct = EXCLUDED.home_win_pct,
            draw_pct = EXCLUDED.draw_pct,
            away_win_pct = EXCLUDED.away_win_pct,
            implied_home = EXCLUDED.implied_home,
            implied_draw = EXCLUDED.implied_draw,
            implied_away = EXCLUDED.implied_away,
            value_home = EXCLUDED.value_home,
            value_draw = EXCLUDED.value_draw,
            value_away = EXCLUDED.value_away,
            home_rating = EXCLUDED.home_rating,
            away_rating = EXCLUDED.away_rating,
            rating_diff = EXCLUDED.rating_diff,
            score = EXCLUDED.score,
            result = EXCLUDED.result,
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at,
            updated_by = EXCLUDED.updated_by;
    """

    rows = []
    for _, r in combined_data.iterrows():
        rows.append([
            r["match_id"],
            r["eci_date"],
            r["oddspedia_date"],
            r["date_diff_days"],
            r["home_team"],
            r["away_team"],
            r["oddspedia_home_team"],
            r["oddspedia_away_team"],
            r["odds_home"],
            r["odds_draw"],
            r["odds_away"],
            r["home_win_pct"],
            r["draw_pct"],
            r["away_win_pct"],
            r["implied_home"],
            r["implied_draw"],
            r["implied_away"],
            r["value_home"],
            r["value_draw"],
            r["value_away"],
            r["home_rating"],
            r["away_rating"],
            r["rating_diff"],
            r["competition"],
            r["oddspedia_competition"],
            r["score"],
            r["result"],
            r["status"],
            r["created_at"],
            r["updated_at"],
            r["created_by"],
            r["updated_by"],
        ])

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, rows)
            conn.commit()

    print(f"Saved {len(rows)} rows to public.{TARGET_TABLE}")


def main():
    print(f"Current UTC: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
    print(f"User: {CURRENT_USER}")

    ensure_target_table()

    df_odd, df_eci = get_data_from_database()
    combined = combine_data(df_odd, df_eci)
    save_to_database(combined)

    print("\nDone.")


if __name__ == "__main__":
    main()