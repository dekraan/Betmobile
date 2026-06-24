# team_matching_eci_oddspedia.py
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import psycopg2
import re
import unicodedata
import logging
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from jellyfish import jaro_winkler_similarity, metaphone, soundex

# ============== CONFIG ==============
CURRENT_USER = "dekraan"
NOW_UTC = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "Betmobile",
    "user": "postgres",
    "password": "300500",
}

# Pad naar je 2-richtingen mapping (uit de builder)
MAPPING_JSON = r"C:\Users\Gebruiker\Documents\Betmobile\team_mappings_eci_oddspedia.json"

# Logging
logging.basicConfig(
    filename=f"team_matching_eci_oddspedia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Interactie-tuning
TOPN_SUGGESTIONS = 7
MIN_SUGGESTION_SCORE = 60  # toon alleen ECI-kandidaten met score >= 60

# ============ Normalisatie & Scoring ============
REMOVABLE_PREFIXES = r'\b(fc|cf|ac|sc|afc|bk|fk|nk|if|sk|ud|cd|sd|ss|ks|ksk|ofk|pfc|cfr)\b'

def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("ASCII").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(REMOVABLE_PREFIXES, " ", s)
    s = " ".join(s.split())
    return s

def clean_team_name(name: str) -> str:
    if not name:
        return ""

    s = str(name).strip()

    # Alleen eerste regel pakken: voorkomt "Arda Kardzhali\nPEN"
    s = s.splitlines()[0].strip()

    # Losse statusrommel weghalen
    bad_tokens = {"PEN", "AET", "FT", "HT"}
    if s.upper() in bad_tokens:
        return ""

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
    weighted = min(100, 0.45*ratio + 0.25*partial + 0.20*last_word + 0.10*jw + contained + phon)
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
    return (f"weighted {s['weighted']} | ratio {s['ratio']}, "
            f"partial {s['partial']}, jw {s['jaro_winkler']}, last {s['last_word']} "
            f"(+{s['contained_bonus']}/+{s['phonetic_bonus']})")

# ============== Kernklasse ==============
class ECIOddMatcher:
    def __init__(self, mapping_json: str):
        self.path = Path(mapping_json)
        self.data = self._load_or_init_json()
        self.undo_stack = []
        self._load_names_from_db()

    # ---- IO ----
    def _load_or_init_json(self):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # zorg dat structuur aanwezig is
            d.setdefault("mapping", {})
            d["mapping"].setdefault("oddspedia_to_eci", {})
            d["mapping"].setdefault("eci_to_oddspedia", {})
            d.setdefault("meta", {})
            return d
        return {
            "meta": {
                "generated_at_utc": NOW_UTC,
                "generated_by": CURRENT_USER,
                "notes": "Oddspedia ↔ ECI mapping only (interactive)."
            },
            "mapping": {
                "oddspedia_to_eci": {},
                "eci_to_oddspedia": {}
            },
            "needs_review": []
        }

    def save(self):
        self.data["meta"]["last_saved_utc"] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        self.data["meta"]["last_saved_by"] = CURRENT_USER
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logging.info("Saved mapping JSON")

    # ---- DB ----
    def _load_names_from_db(self):
        with psycopg2.connect(**DB_CONFIG) as conn:
            odd = pd.read_sql("SELECT home_team, away_team FROM oddspedia_unibet_backbone", conn)
            eci = pd.read_sql("SELECT home_team, away_team FROM eci_data", conn)
        odd_names = set(odd["home_team"]).union(odd["away_team"])
        eci_names = set(eci["home_team"]).union(eci["away_team"])

        self.odd_names = sorted({
            clean_team_name(x)
            for x in odd_names
            if clean_team_name(x)
        })

        self.eci_names = sorted({
            clean_team_name(x)
            for x in eci_names
            if clean_team_name(x)
        })

    def unmapped_odd(self):
        mapped = set(clean_team_name(x) for x in self.data["mapping"]["oddspedia_to_eci"].keys())
        eci_exact = set(clean_team_name(x) for x in self.eci_names)

        remaining = []

        for t in self.odd_names:
            clean = clean_team_name(t)

            if not clean:
                continue

            # staat al in mapping-json
            if clean in mapped:
                continue

            # bestaat exact als ECI-teamnaam
            if clean in eci_exact:
                continue

            remaining.append(clean)

        return sorted(set(remaining))

    # ---- Suggesties & zoeken ----
    def suggestions_for_odd(self, odd_name: str, topn=TOPN_SUGGESTIONS, min_score=MIN_SUGGESTION_SCORE):
        out = []
        for eci in self.eci_names:
            s = score_pair(odd_name, eci)
            if s["weighted"] >= min_score:
                out.append((eci, s))
        out.sort(key=lambda x: x[1]["weighted"], reverse=True)
        return out[:topn]

    def search_eci(self, query: str, topn=15):
        qn = normalize_name(query)
        rows = []
        for eci in self.eci_names:
            s = score_pair(qn, eci)  # score t.o.v. genormaliseerde query
            rows.append((eci, s))
        rows.sort(key=lambda x: x[1]["weighted"], reverse=True)
        return rows[:topn]

    # ---- Mappen + undo ----
    def set_pair(self, odd_name: str, eci_name: str):
        odd_name = clean_team_name(odd_name)
        eci_name = clean_team_name(eci_name)

        prev_odd2eci = self.data["mapping"]["oddspedia_to_eci"].get(odd_name)
        prev_eci2odd = self.data["mapping"]["eci_to_oddspedia"].get(eci_name)

        self.undo_stack.append(("set", odd_name, prev_odd2eci, eci_name, prev_eci2odd))

        # Belangrijk:
        # meerdere Oddspedia-namen mogen naar dezelfde ECI-naam wijzen.
        self.data["mapping"]["oddspedia_to_eci"][odd_name] = eci_name

        # Reverse is alleen informatief; niet meer gebruiken als harde unieke koppeling.
        self.data["mapping"]["eci_to_oddspedia"][eci_name] = odd_name

    def undo(self):
        if not self.undo_stack:
            return "Nothing to undo."
        action, odd_name, prev_odd2eci, eci_name, prev_eci2odd = self.undo_stack.pop()
        if action == "set":
            # herstel odd->eci
            if prev_odd2eci is None:
                self.data["mapping"]["oddspedia_to_eci"].pop(odd_name, None)
            else:
                self.data["mapping"]["oddspedia_to_eci"][odd_name] = prev_odd2eci
            # herstel eci->odd
            if prev_eci2odd is None:
                self.data["mapping"]["eci_to_oddspedia"].pop(eci_name, None)
            else:
                self.data["mapping"]["eci_to_oddspedia"][eci_name] = prev_eci2odd
            return f"Undid mapping {odd_name} ↔ {eci_name}"
        return "Unknown undo action."

    # ---- Stats ----
    def stats_line(self):
        return (f"Oddspedia teams: {len(self.odd_names)} "
                f"(mapped {len(self.data['mapping']['oddspedia_to_eci'])}) | "
                f"ECI teams: {len(self.eci_names)} "
                f"(mapped {len(self.data['mapping']['eci_to_oddspedia'])})")

# ============== Interactieve loop ==============
def interactive_loop():
    m = ECIOddMatcher(MAPPING_JSON)
    print("ECI ↔ Oddspedia Team Matcher (interactive)")
    print(f"UTC: {NOW_UTC} | user: {CURRENT_USER}")
    remaining_now = m.unmapped_odd()

    print(m.stats_line())
    print(f"Nog handmatig te beoordelen: {len(remaining_now)}")

    if remaining_now:
        print("Eerste 25:")
        for name in remaining_now[:25]:
            print(f" - {name}")
    logging.info("Interactive matcher started")

    # >>> NIEUW: sessie-skips (niet persistent)
    skipped_session = set()

    idx = 0
    while True:
        # >>> NIEUW: filter de sessie-skips weg
        remaining = [t for t in m.unmapped_odd() if t not in skipped_session]

        print(f"\nNog over deze sessie: {len(remaining)}")
        if not remaining:
            print("\n🎉 Geen onbehepte Oddspedia-teams meer. Gereed!")
            m.save()
            break

        remaining.sort()
        odd = remaining[0]
        idx += 1
        print("\n" + "="*70)
        print(f"[{idx}] Oddspedia: {odd}")
        suggs = m.suggestions_for_odd(odd)

        if suggs:
            print("\nSuggesties (ECI):")
            for i, (eci, sc) in enumerate(suggs, 1):
                print(f"  {i}. {eci}   [{score_str(sc)}]")
        else:
            print("\nGeen automatische suggesties ≥", MIN_SUGGESTION_SCORE)

        print("\nKies actie:")
        print("  - Typ een nummer (1..N) om die suggestie te kiezen")
        print("  - s = zoeken in ECI")
        print("  - k = skip deze Oddspedia-naam")
        print("  - u = undo laatste mapping")
        print("  - q = save & quit")
        choice = input("> ").strip().lower()

        if choice == "q":
            m.save()
            print("Progress opgeslagen. Stoppen.")
            break

        elif choice == "u":
            print(m.undo())
            m.save()
            continue

        elif choice == "k":
            # >>> NIEUW: zet deze naam in de sessie-skipset
            skipped_session.add(odd)
            print("Overgeslagen (alleen deze sessie). Volgende team.")
            continue

        elif choice == "s":
            query = input("Zoekterm in ECI: ").strip()
            if not query:
                continue
            results = m.search_eci(query)
            if not results:
                print("Geen hits.")
                continue
            print("\nZoekresultaten:")
            for i, (eci, sc) in enumerate(results, 1):
                print(f"  {i}. {eci}   [{score_str(sc)}]")
            pick = input("Kies nummer (of Enter om te annuleren): ").strip()
            if pick.isdigit():
                pi = int(pick)
                if 1 <= pi <= len(results):
                    eci_pick = results[pi-1][0]
                    m.set_pair(odd, eci_pick)
                    m.save()
                    print(f"✔ Gekoppeld: {odd} → {eci_pick}")
            continue

        elif choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(suggs):
                eci_pick = suggs[num-1][0]
                m.set_pair(odd, eci_pick)
                m.save()
                print(f"✔ Gekoppeld: {odd} → {eci_pick}")
            else:
                print("Ongeldig nummer.")
            continue

        else:
            print("Onbekende keuze.")
            continue


if __name__ == "__main__":
    print("ECI ↔ Oddspedia Team Matcher — starting…")
    interactive_loop()
