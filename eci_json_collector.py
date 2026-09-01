"""
eci_json_collector.py
=====================

Append-only collector voor de ECI REST-feed, met ingebouwde diff-analyse.

Draait NAAST je bestaande eci_scraper.py. Schrijft niets naar PostgreSQL en
raakt geen bestaande bestanden aan. Alles gaat naar losse JSON-bestanden.

WAAROM
------
De feed levert per wedstrijd twee ratings per team:

    HomeECI            1683.16
    HomeRankingPoints  1641.68

Vermoeden: HomeECI is de rating zoals die gold ten tijde van de wedstrijd,
RankingPoints is de huidige ranglijstwaarde. Als dat klopt zit er historie in
de feed en hoef je niet tot oktober te wachten op eigen snapshots.

De diff-modus toetst dat: als HomeECI bij AFGESPEELDE wedstrijden stil blijft
staan terwijl RankingPoints beweegt, is het vermoeden bevestigd.

GEBRUIK
-------
    pip install requests

    python eci_json_collector.py collect
    python eci_json_collector.py collect --leagues 2,52,656

    python eci_json_collector.py runs

    (twee dagen later)
    python eci_json_collector.py diff
    python eci_json_collector.py diff --run-a run_20260831T1030Z --run-b run_20260902T1030Z

Output staat in ./eci_snapshots/ naast dit script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests ontbreekt. Installeer met: pip install requests")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = "https://www.euroclubindex.com"
REST_URL = f"{BASE}/wp-json/happyhorizon/v1/get-module-match-odds/"
PAGE_URL = f"{BASE}/match-odds/"

ROOT = Path(__file__).resolve().parent / "eci_snapshots"

# Pauze tussen requests. Niet lager zetten.
DELAY_SECONDS = 1.5
TIMEOUT = 30
RETRIES = 2

# Velden waarvan we de waarde vergelijken in de diff.
NUMERIC_FIELDS = (
    "HomeECI",
    "AwayECI",
    "HomeRankingPoints",
    "AwayRankingPoints",
    "n_OddHomeWin",
    "n_OddDraw",
    "n_OddAwayWin",
)

# Verschil kleiner dan dit telt als "niet veranderd" (float-ruis).
EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%MZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": PAGE_URL,
        }
    )
    return s


def fetch_json(session: requests.Session, params: dict | None):
    """Haal JSON op met een paar retries. Geeft (data, headers) of (None, None)."""
    last_error = None
    for attempt in range(1, RETRIES + 1):
        time.sleep(DELAY_SECONDS)
        try:
            resp = session.get(REST_URL, params=params, timeout=TIMEOUT)
            if resp.status_code != 200:
                last_error = f"status {resp.status_code}"
                continue
            return resp.json(), dict(resp.headers)
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        if attempt < RETRIES:
            time.sleep(DELAY_SECONDS * 2)
    print(f"      [--] mislukt: {last_error}")
    return None, None


def to_float(value):
    """De feed levert alles als string. Expliciet casten, nooit impliciet."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def get_league_index(session: requests.Session) -> list[dict]:
    data, _ = fetch_json(session, None)
    if not data or "items" not in data:
        sys.exit("Kon de competitielijst niet ophalen.")

    leagues = []
    for item in data.get("items", []):
        cd = item.get("competitionData") or {}
        cid = cd.get("n_CompetitionID")
        if cid:
            leagues.append(
                {
                    "n_CompetitionID": str(cid),
                    "c_Competition": cd.get("c_Competition"),
                    "c_CompetitionNatio": cd.get("c_CompetitionNatio"),
                    "c_CompetitionNatioShort": cd.get("c_CompetitionNatioShort"),
                    "n_CompetitionLevel": cd.get("n_CompetitionLevel"),
                    "c_CompetitionType": cd.get("c_CompetitionType"),
                }
            )
    return leagues


def collect(only_leagues: list[str] | None) -> None:
    session = make_session()
    rid = run_id()
    outdir = ROOT / rid
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {rid}")
    print(f"Map: {outdir}\n")

    leagues = get_league_index(session)
    (outdir / "leagues.json").write_text(
        json.dumps(leagues, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] {len(leagues)} competities in de index")

    if only_leagues:
        wanted = set(only_leagues)
        leagues = [l for l in leagues if l["n_CompetitionID"] in wanted]
        missing = wanted - {l["n_CompetitionID"] for l in leagues}
        if missing:
            print(f"[--] onbekende competitie-ID's overgeslagen: {sorted(missing)}")

    print(f"[--] ophalen van {len(leagues)} competities "
          f"(~{int(len(leagues) * DELAY_SECONDS)}s)\n")

    records = []
    failures = []

    for i, league in enumerate(leagues, start=1):
        cid = league["n_CompetitionID"]
        label = f"{league['c_CompetitionNatioShort']} {league['c_Competition']}"
        print(f"  [{i:>2}/{len(leagues)}] {cid:>5}  {label[:45]}", end="")

        data, _ = fetch_json(session, {"selected_league": cid})
        if data is None:
            failures.append(cid)
            continue

        # Ruwe respons altijd bewaren: onbewerkt is onvervangbaar.
        (rawdir / f"{cid}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

        odds = data.get("matchOdds") or []
        fetched = utc_now_iso()
        for row in odds:
            row = dict(row)
            # Deze twee staan niet in het record zelf.
            row["n_CompetitionID"] = cid
            row["fetched_at_utc"] = fetched
            records.append(row)

        print(f"  -> {len(odds)} wedstrijden")

    # Alles plat in een JSONL, makkelijk in te lezen met pandas.
    jsonl = outdir / "matches.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "run_id": rid,
        "collected_at_utc": utc_now_iso(),
        "leagues_requested": len(leagues),
        "leagues_failed": failures,
        "match_records": len(records),
        "unique_match_ids": len({r.get("n_MatchID") for r in records}),
    }
    (outdir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[OK] {len(records)} wedstrijdrecords "
          f"({meta['unique_match_ids']} unieke n_MatchID)")
    if failures:
        print(f"[--] mislukte competities: {failures}")
    print(f"[OK] weggeschreven naar {jsonl}")


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def list_runs() -> list[Path]:
    if not ROOT.exists():
        return []
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("run_"))


def show_runs() -> None:
    runs = list_runs()
    if not runs:
        print("Nog geen runs. Draai eerst: python eci_json_collector.py collect")
        return
    print(f"{len(runs)} run(s):\n")
    for r in runs:
        meta_path = r / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"  {r.name}   {meta.get('match_records', '?'):>6} records"
                  f"   {meta.get('collected_at_utc', '?')}")
        else:
            print(f"  {r.name}   (geen meta.json)")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def load_matches(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "matches.jsonl"
    if not path.exists():
        sys.exit(f"Geen matches.jsonl in {run_dir}")
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mid = row.get("n_MatchID")
            if mid:
                out[str(mid)] = row
    return out


def diff(run_a: str | None, run_b: str | None) -> None:
    runs = list_runs()
    if len(runs) < 2:
        sys.exit("Minstens twee runs nodig. Draai 'collect' nog een keer over een paar dagen.")

    dir_a = ROOT / run_a if run_a else runs[0]
    dir_b = ROOT / run_b if run_b else runs[-1]

    print(f"Vergelijking\n  A: {dir_a.name}\n  B: {dir_b.name}\n")

    a = load_matches(dir_a)
    b = load_matches(dir_b)

    shared = sorted(set(a) & set(b))
    print(f"  in A: {len(a)}   in B: {len(b)}   gedeeld: {len(shared)}")
    print(f"  alleen in A: {len(set(a) - set(b))}   alleen in B: {len(set(b) - set(a))}\n")

    if not shared:
        sys.exit("Geen gedeelde wedstrijden. Zijn dit dezelfde competities?")

    # Splitsen op afgespeeld / nog te spelen: daar zit de hele test in.
    groups = {"afgespeeld (hasResults=1)": [], "nog te spelen (hasResults=0)": []}
    for mid in shared:
        key = ("afgespeeld (hasResults=1)"
               if str(a[mid].get("hasResults")) == "1"
               else "nog te spelen (hasResults=0)")
        groups[key].append(mid)

    for group_name, ids in groups.items():
        print(f"--- {group_name}: {len(ids)} wedstrijden ---")
        if not ids:
            print()
            continue

        for field in NUMERIC_FIELDS:
            deltas = []
            missing = 0
            for mid in ids:
                va, vb = to_float(a[mid].get(field)), to_float(b[mid].get(field))
                if va is None or vb is None:
                    missing += 1
                    continue
                deltas.append(vb - va)

            if not deltas:
                print(f"  {field:<20} geen bruikbare waarden ({missing} ontbrekend)")
                continue

            changed = [d for d in deltas if abs(d) > EPSILON]
            pct = 100.0 * len(changed) / len(deltas)
            if changed:
                mean_abs = statistics.fmean(abs(d) for d in changed)
                max_abs = max(abs(d) for d in changed)
                print(f"  {field:<20} {len(changed):>4}/{len(deltas):<4} veranderd "
                      f"({pct:5.1f}%)  gem |delta| {mean_abs:.4f}  max {max_abs:.4f}")
            else:
                print(f"  {field:<20} {0:>4}/{len(deltas):<4} veranderd "
                      f"(  0.0%)  volledig stabiel")
        print()

    print("HOE TE LEZEN")
    print("  Bij AFGESPEELDE wedstrijden:")
    print("    HomeECI stabiel + RankingPoints beweegt")
    print("      -> vermoeden bevestigd: ECI is historisch, feed bevat ratinghistorie.")
    print("    beide stabiel")
    print("      -> allebei bevroren na afloop; geen historie te winnen.")
    print("    beide bewegen")
    print("      -> iets anders aan de hand; check de methodology-pagina.")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ECI JSON collector en diff")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="haal alle competities op en bewaar")
    p_collect.add_argument(
        "--leagues",
        help="komma-gescheiden n_CompetitionID's, bv. 2,52,656 (default: alle)",
    )

    sub.add_parser("runs", help="toon opgeslagen runs")

    p_diff = sub.add_parser("diff", help="vergelijk twee runs op n_MatchID")
    p_diff.add_argument("--run-a", help="oudste run (default: eerste)")
    p_diff.add_argument("--run-b", help="nieuwste run (default: laatste)")

    args = parser.parse_args()

    if args.command == "collect":
        only = [x.strip() for x in args.leagues.split(",")] if args.leagues else None
        collect(only)
    elif args.command == "runs":
        show_runs()
    elif args.command == "diff":
        diff(args.run_a, args.run_b)


if __name__ == "__main__":
    main()