import pandas as pd
import psycopg2
import re
import unicodedata
import logging
from datetime import datetime, timezone
from fuzzywuzzy import fuzz
from jellyfish import jaro_winkler_similarity, metaphone, soundex

# ============== CONFIG ==============
CURRENT_USER = "dekraan"
NOW_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

from betmobile_settings import DB_CONFIG

LOGFILE = f"team_matching_eci_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

TOPN_SUGGESTIONS = 7
MIN_SUGGESTION_SCORE = 60

REMOVABLE_PREFIXES = r"\b(fc|cf|ac|sc|afc|bk|fk|nk|if|sk|ud|cd|sd|ss|ks|ksk|ofk|pfc|cfr)\b"

# ============ Normalisatie & Scoring ============
def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("ASCII").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(REMOVABLE_PREFIXES, " ", s)
    s = " ".join(s.split())
    return s


def score_pair(a: str, b: str) -> dict:
    n1, n2 = normalize_name(a), normalize_name(b)

    partial = fuzz.partial_ratio(n1, n2)
    ratio = fuzz.ratio(n1, n2)
    jw = int(jaro_winkler_similarity(n1, n2) * 100)

    lw1 = n1.split()[-1] if n1.split() else ""
    lw2 = n2.split()[-1] if n2.split() else ""
    last_word = fuzz.ratio(lw1, lw2)

    contained = 20 if (n1 in n2 or n2 in n1) else 0
    phon = 15 if (metaphone(n1) == metaphone(n2) or soundex(n1) == soundex(n2)) else 0

    weighted = min(100, 0.45 * ratio + 0.25 * partial + 0.20 * last_word + 0.10 * jw + contained + phon)

    return {
        "ratio": ratio,
        "partial": partial,
        "jaro_winkler": jw,
        "last_word": last_word,
        "contained_bonus": contained,
        "phonetic_bonus": phon,
        "weighted": round(weighted, 1),
    }


def score_str(s: dict) -> str:
    return (
        f"weighted {s['weighted']} | ratio {s['ratio']}, "
        f"partial {s['partial']}, jw {s['jaro_winkler']}, last {s['last_word']} "
        f"(+{s['contained_bonus']}/+{s['phonetic_bonus']})"
    )


# ============== Kernklasse ==============
class ECIAPIMatcher:
    def __init__(self):
        self.undo_stack = []
        self._load_names_from_db()

    def _get_conn(self):
        return psycopg2.connect(**DB_CONFIG)

    def _load_names_from_db(self):
        with self._get_conn() as conn:
            # ECI teams + competitie + api_league_id via league_aliases
            self.eci_df = pd.read_sql(
                """
                WITH eci_teams AS (
                    SELECT DISTINCT
                        trim(ed.home_team) AS eci_team_name,
                        ed.competition
                    FROM eci_data ed

                    UNION

                    SELECT DISTINCT
                        trim(ed.away_team) AS eci_team_name,
                        ed.competition
                    FROM eci_data ed
                )
                SELECT
                    e.eci_team_name,
                    e.competition,
                    la.api_league_id
                FROM eci_teams e
                JOIN league_aliases la
                  ON la.eci_key = e.competition
                ORDER BY e.competition, e.eci_team_name
                """,
                conn,
            )

            # API teams alleen binnen leagues die in league_aliases staan
            self.api_df = pd.read_sql(
                """
                SELECT DISTINCT
                    f.league_id AS api_league_id,
                    t.team_id AS api_team_id,
                    trim(t.name) AS api_team_name,
                    l.country AS api_country
                FROM fixtures f
                JOIN teams t
                  ON t.team_id = f.home_team_id
                  OR t.team_id = f.away_team_id
                JOIN leagues l
                  ON l.league_id = f.league_id
                JOIN league_aliases la
                  ON la.api_league_id = f.league_id
                ORDER BY api_league_id, api_team_name
                """,
                conn,
            )

            # Al gematchte ECI namen
            matched = pd.read_sql(
                """
                SELECT DISTINCT eci_team_name
                FROM team_aliases
                WHERE is_active = true
                """,
                conn,
            )

        self.matched_eci_names = set(matched["eci_team_name"].tolist())
        logging.info("Loaded ECI/API data from DB")

    def reload(self):
        self._load_names_from_db()

    def unmapped_eci(self):
        df = self.eci_df[~self.eci_df["eci_team_name"].isin(self.matched_eci_names)].copy()
        df = df.drop_duplicates(subset=["eci_team_name", "competition", "api_league_id"])
        return df.sort_values(["competition", "eci_team_name"]).reset_index(drop=True)

    def suggestions_for_eci(self, eci_name: str, api_league_id: int, topn=TOPN_SUGGESTIONS, min_score=MIN_SUGGESTION_SCORE):
        out = []

        candidates = self.api_df[self.api_df["api_league_id"] == api_league_id].copy()

        for _, row in candidates.iterrows():
            api_name = row["api_team_name"]
            api_id = int(row["api_team_id"])
            api_country = row["api_country"]

            s = score_pair(eci_name, api_name)

            if s["weighted"] >= min_score:
                out.append((api_id, api_name, api_country, s))

        out.sort(key=lambda x: x[3]["weighted"], reverse=True)
        return out[:topn]

    def search_api(self, query: str, api_league_id: int, topn=15):
        qn = normalize_name(query)
        rows = []

        candidates = self.api_df[self.api_df["api_league_id"] == api_league_id].copy()

        for _, row in candidates.iterrows():
            api_name = row["api_team_name"]
            api_id = int(row["api_team_id"])
            api_country = row["api_country"]

            s = score_pair(qn, api_name)
            rows.append((api_id, api_name, api_country, s))

        rows.sort(key=lambda x: x[3]["weighted"], reverse=True)
        return rows[:topn]

    def save_match(self, eci_name: str, api_id: int, api_name: str, api_country: str, api_league_id: int):
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO team_aliases (
                        eci_team_name,
                        api_team_id,
                        api_team_name,
                        api_country,
                        api_league_id,
                        match_method,
                        is_active,
                        notes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, true, %s)
                    ON CONFLICT (eci_team_name, api_team_id) DO NOTHING
                    RETURNING alias_id
                    """,
                    (
                        eci_name,
                        api_id,
                        api_name,
                        api_country,
                        api_league_id,
                        "manual",
                        f"matched by {CURRENT_USER} at {NOW_UTC}",
                    ),
                )
                row = cur.fetchone()
                conn.commit()

        self.undo_stack.append((eci_name, api_id))
        self.matched_eci_names.add(eci_name)
        logging.info(f"Saved match: {eci_name} -> {api_name} ({api_id})")

        return row[0] if row else None

    def undo(self):
        if not self.undo_stack:
            return "Nothing to undo."

        eci_name, api_id = self.undo_stack.pop()

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM team_aliases
                    WHERE eci_team_name = %s
                      AND api_team_id = %s
                    """,
                    (eci_name, api_id),
                )
                deleted = cur.rowcount
                conn.commit()

        self.reload()

        if deleted:
            logging.info(f"Undo deleted match: {eci_name} -> api_team_id={api_id}")
            return f"Undid mapping {eci_name} ↔ api_team_id {api_id}"
        return "Nothing deleted."

    def stats_line(self):
        total_eci = self.eci_df["eci_team_name"].nunique()
        mapped_eci = len(self.matched_eci_names)
        total_api = self.api_df["api_team_id"].nunique()

        return (
            f"ECI teams: {total_eci} (mapped {mapped_eci}) | "
            f"API teams in mapped leagues: {total_api}"
        )


# ============== Interactieve loop ==============
def interactive_loop():
    m = ECIAPIMatcher()

    print("ECI ↔ API-Football Team Matcher (interactive)")
    print(f"UTC: {NOW_UTC} | user: {CURRENT_USER}")
    print(m.stats_line())
    print(f"Logfile: {LOGFILE}")
    logging.info("Interactive matcher started")

    skipped_session = set()
    idx = 0

    while True:
        remaining_df = m.unmapped_eci()
        remaining_df = remaining_df[~remaining_df["eci_team_name"].isin(skipped_session)]

        if remaining_df.empty:
            print("\n🎉 Geen unmatched ECI-teams meer in deze sessie.")
            break

        row = remaining_df.iloc[0]
        eci_name = row["eci_team_name"]
        competition = row["competition"]
        api_league_id = int(row["api_league_id"])

        idx += 1

        print("\n" + "=" * 80)
        print(f"[{idx}] ECI team : {eci_name}")
        print(f"    Comp     : {competition}")
        print(f"    API league_id : {api_league_id}")

        suggs = m.suggestions_for_eci(eci_name, api_league_id)

        if suggs:
            print("\nSuggesties (API teams):")
            for i, (api_id, api_name, api_country, sc) in enumerate(suggs, 1):
                print(f"  {i}. {api_name} [id={api_id}, country={api_country}]   [{score_str(sc)}]")
        else:
            print(f"\nGeen automatische suggesties ≥ {MIN_SUGGESTION_SCORE}")

        print("\nKies actie:")
        print("  - Typ een nummer (1..N) om die suggestie te kiezen")
        print("  - s = zoeken in API-teams binnen dezelfde league")
        print("  - k = skip deze ECI-naam (alleen deze sessie)")
        print("  - u = undo laatste mapping")
        print("  - r = refresh data uit database")
        print("  - q = quit")

        choice = input("> ").strip().lower()

        if choice == "q":
            print("Stoppen.")
            break

        elif choice == "u":
            print(m.undo())
            continue

        elif choice == "r":
            m.reload()
            print("Data opnieuw geladen.")
            continue

        elif choice == "k":
            skipped_session.add(eci_name)
            print("Overgeslagen voor deze sessie.")
            continue

        elif choice == "s":
            query = input("Zoekterm in API-teams: ").strip()
            if not query:
                continue

            results = m.search_api(query, api_league_id)

            if not results:
                print("Geen hits.")
                continue

            print("\nZoekresultaten (API):")
            for i, (api_id, api_name, api_country, sc) in enumerate(results, 1):
                print(f"  {i}. {api_name} [id={api_id}, country={api_country}]   [{score_str(sc)}]")

            pick = input("Kies nummer (of Enter om te annuleren): ").strip()

            if pick.isdigit():
                pi = int(pick)
                if 1 <= pi <= len(results):
                    api_id, api_name, api_country, _ = results[pi - 1]
                    m.save_match(eci_name, api_id, api_name, api_country, api_league_id)
                    print(f"✔ Gekoppeld: {eci_name} → {api_name}")
                else:
                    print("Ongeldig nummer.")
            continue

        elif choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(suggs):
                api_id, api_name, api_country, _ = suggs[num - 1]
                m.save_match(eci_name, api_id, api_name, api_country, api_league_id)
                print(f"✔ Gekoppeld: {eci_name} → {api_name}")
            else:
                print("Ongeldig nummer.")
            continue

        else:
            print("Onbekende keuze.")
            continue


if __name__ == "__main__":
    print("ECI ↔ API-Football Team Matcher — starting…")
    interactive_loop()