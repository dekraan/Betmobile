"""
eci_history_collector.py
========================

Haalt de volledige ratinghistorie per team op uit het team-details endpoint
en controleert eerst of die historie hetzelfde meet als de match-odds feed.

    .../wp-json/happyhorizon/v1/get-module-team-details/?selected_team=<ECI team-ID>

Geeft ~1000 punten per team, wekelijks, van 2007 tot nu, met Date, Points
en Ranking. Volledige precisie.

Schrijft niets naar PostgreSQL.

Gebruik
-------
    python eci_history_collector.py validate
    python eci_history_collector.py validate --teams 10

    python eci_history_collector.py collect
    python eci_history_collector.py collect --limit 20

Historie verandert niet met terugwerkende kracht, dus collect hoef je in
principe maar een keer te draaien. Onderbreken mag: bij een herstart worden
al opgehaalde teams overgeslagen.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests ontbreekt. Installeer met: pip install requests")


BASE = "https://www.euroclubindex.com"
REST_URL = f"{BASE}/wp-json/happyhorizon/v1/get-module-team-details/"
RANKING_URL = f"{BASE}/wp-json/happyhorizon/v1/get-ranking"
PAGE_URL = f"{BASE}/teams/arsenal/"

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "eci_history"
RAWDIR = OUTDIR / "raw"
SNAPSHOTS = HERE / "eci_snapshots"

DELAY = 1.5
TIMEOUT = 30
RETRIES = 2

# Binnen een halve punt = dezelfde waarde, gegeven afronding elders.
TOLERANCE = 0.5


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Referer": PAGE_URL,
    })
    return s


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_eci_date(value: str):
    """De feed gebruikt 'Jul 2, 2007'."""
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


def extract_series(data) -> list[dict]:
    """Haal de reeks uit graphData.teamRankings, met een terugval voor het
    geval de structuur ooit verandert."""
    if isinstance(data, dict):
        graph = data.get("graphData")
        if isinstance(graph, dict):
            rankings = graph.get("teamRankings")
            if isinstance(rankings, list) and rankings:
                return rankings

    # Terugval: zoek de langste lijst van dicts met Date en Points.
    best: list = []

    def walk(node, depth=0):
        nonlocal best
        if depth > 8:
            return
        if isinstance(node, list) and node and isinstance(node[0], dict):
            keys = {k.lower() for k in node[0]}
            if "date" in keys and ("points" in keys or "eci" in keys):
                if len(node) > len(best):
                    best = node
        if isinstance(node, dict):
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, depth + 1)

    walk(data)
    return best


def fetch_team(session: requests.Session, team_id: str):
    """Geeft (ruwe json, genormaliseerde reeks) of (None, None)."""
    last_error = None
    for attempt in range(1, RETRIES + 1):
        time.sleep(DELAY)
        try:
            resp = session.get(REST_URL, params={"selected_team": team_id},
                               timeout=TIMEOUT)
            if resp.status_code != 200:
                last_error = f"status {resp.status_code}"
                continue
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            data = None
        if attempt < RETRIES:
            time.sleep(DELAY * 2)
    else:
        print(f"      [--] {team_id}: {last_error}")
        return None, None

    if data is None:
        print(f"      [--] {team_id}: {last_error}")
        return None, None

    series = []
    for row in extract_series(data):
        d = parse_eci_date(row.get("Date"))
        p = to_float(row.get("Points"))
        if d is None or p is None:
            continue
        series.append({
            "eci_team_id": str(team_id),
            "date": d.isoformat(),
            "points": p,
            "ranking": to_float(row.get("Ranking")),
        })
    series.sort(key=lambda r: r["date"])
    return data, series


# ---------------------------------------------------------------------------
# Teams uit de match-odds feed
# ---------------------------------------------------------------------------

def latest_run() -> Path:
    if not SNAPSHOTS.exists():
        sys.exit("Geen eci_snapshots gevonden. Draai eerst eci_json_collector.py collect")
    runs = sorted(p for p in SNAPSHOTS.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    if not runs:
        sys.exit("Geen runs in eci_snapshots.")
    return runs[-1]


def load_feed_matches(run_dir: Path) -> list[dict]:
    path = run_dir / "matches.jsonl"
    if not path.exists():
        sys.exit(f"Geen matches.jsonl in {run_dir}")
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def teams_from_ranking(session: requests.Session) -> dict[str, str]:
    """Volledige teamlijst uit get-ranking: ~1291 teams met ECI-team-ID.

    Ruimer dan de wedstrijdfeed, die alleen teams bevat die deze weken
    spelen. Bevat ook Points, PrevPoints, Rank en PrevRank; die bewaren we
    apart als momentopname.
    """
    time.sleep(DELAY)
    try:
        resp = session.get(RANKING_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[--] get-ranking mislukt: {exc}")
        return {}

    if not isinstance(data, list):
        print("[--] onverwachte structuur van get-ranking")
        return {}

    OUTDIR.mkdir(exist_ok=True)
    (OUTDIR / "ranking_snapshot.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    teams = {}
    for row in data:
        tid = row.get("teamID")
        if tid:
            teams[str(tid)] = row.get("teamName") or "?"

    print(f"[OK] {len(teams):,} teams uit get-ranking "
          f"(momentopname bewaard in ranking_snapshot.json)")
    return teams


def teams_from_feed(matches: list[dict]) -> dict[str, str]:
    """ECI team-ID -> teamnaam, voor de log."""
    teams = {}
    for m in matches:
        for id_key, name_key in (("n_HomeTeamID", "c_HomeTeam"),
                                 ("n_AwayTeamID", "c_Awayteam")):
            tid = m.get(id_key)
            if tid:
                teams.setdefault(str(tid), m.get(name_key) or "?")
    return teams


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def value_on_or_before(series: list[dict], target: str):
    """Laatste punt op of voor de wedstrijddatum. Dat is wat gold bij aftrap."""
    best = None
    for row in series:
        if row["date"] <= target:
            best = row
        else:
            break
    return best


def nearest_value(series: list[dict], target: str):
    if not series:
        return None
    t = datetime.fromisoformat(target).date()
    return min(series,
               key=lambda r: abs((datetime.fromisoformat(r["date"]).date() - t).days))


def validate(n_teams: int) -> None:
    run_dir = latest_run()
    matches = load_feed_matches(run_dir)
    teams = teams_from_feed(matches)

    print(f"Feed: {run_dir.name}, {len(matches):,} wedstrijden, {len(teams):,} teams\n")

    # Alleen teams met afgespeelde wedstrijden: daar is ECI bevroren en
    # dus vergelijkbaar met een historisch punt.
    played = [m for m in matches if str(m.get("hasResults")) == "1"]
    candidates = sorted({str(m["n_HomeTeamID"]) for m in played if m.get("n_HomeTeamID")})
    if not candidates:
        sys.exit("Geen afgespeelde wedstrijden in de feed.")

    random.seed(20260831)
    sample = random.sample(candidates, min(n_teams, len(candidates)))

    session = make_session()
    OUTDIR.mkdir(exist_ok=True)

    rows_before, rows_nearest = [], []

    for i, tid in enumerate(sample, start=1):
        name = teams.get(tid, "?")
        print(f"  [{i}/{len(sample)}] {tid:>6}  {name[:34]:<34}", end="")

        _, series = fetch_team(session, tid)
        if not series:
            print("  geen historie")
            continue
        print(f"  {len(series):>5} punten  {series[0]['date']} .. {series[-1]['date']}")

        for m in played:
            match_date = str(m.get("d_Date", ""))[:10]
            for id_key, eci_key in (("n_HomeTeamID", "HomeECI"),
                                    ("n_AwayTeamID", "AwayECI")):
                if str(m.get(id_key)) != tid:
                    continue
                feed_value = to_float(m.get(eci_key))
                if feed_value is None:
                    continue

                hit = value_on_or_before(series, match_date)
                if hit:
                    rows_before.append((abs(hit["points"] - feed_value),
                                        match_date, hit["date"], feed_value, hit["points"]))
                near = nearest_value(series, match_date)
                if near:
                    rows_nearest.append(abs(near["points"] - feed_value))

    if not rows_before:
        print("\nGeen vergelijkingen gemaakt.")
        return

    print(f"\n{'=' * 72}\nVALIDATIE: historie versus match-odds feed\n{'=' * 72}")

    for label, values in (
        ("laatste punt op of voor de wedstrijd", [r[0] for r in rows_before]),
        ("dichtstbijzijnde punt", rows_nearest),
    ):
        n = len(values)
        within = sum(1 for v in values if v <= TOLERANCE)
        mean = sum(values) / n
        print(f"\n  {label}")
        print(f"    n = {n:,}")
        print(f"    binnen {TOLERANCE}: {within:,} ({100.0 * within / n:.1f}%)")
        print(f"    gemiddeld verschil: {mean:.3f}")
        print(f"    grootste verschil:  {max(values):.3f}")

    worst = sorted(rows_before, key=lambda r: -r[0])[:5]
    print("\n  grootste afwijkingen (op of voor):")
    for diff, md, hd, feed_v, hist_v in worst:
        print(f"    wedstrijd {md}  historiepunt {hd}  "
              f"feed {feed_v:.2f}  historie {hist_v:.2f}  verschil {diff:.2f}")

    print("\n  HOE TE LEZEN")
    print("    hoog percentage binnen 0.5  -> beide bronnen meten hetzelfde,")
    print("                                   historie is bruikbaar")
    print("    structureel afwijkend       -> andere grootheid of ander")
    print("                                   peilmoment; niet combineren")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def collect(limit: int | None, source: str) -> None:
    OUTDIR.mkdir(exist_ok=True)
    RAWDIR.mkdir(exist_ok=True)

    session = make_session()

    if source == "ranking":
        teams = teams_from_ranking(session)
        if not teams:
            print("[--] terugval op de wedstrijdfeed")
            teams = teams_from_feed(load_feed_matches(latest_run()))
    else:
        teams = teams_from_feed(load_feed_matches(latest_run()))
        print(f"[OK] {len(teams):,} teams uit de wedstrijdfeed")

    todo = sorted(teams, key=lambda t: int(t) if t.isdigit() else 0)
    if limit:
        todo = todo[:limit]

    # Hervatten: al opgehaalde teams overslaan.
    done = {p.stem for p in RAWDIR.glob("*.json")}
    remaining = [t for t in todo if t not in done]

    print(f"teams in de lijst: {len(teams):,}")
    print(f"al opgehaald: {len(done):,}")
    print(f"nog te doen: {len(remaining):,}  "
          f"(~{int(len(remaining) * DELAY / 60)} minuten)\n")

    if not remaining:
        print("Niets te doen.")
    failures = []

    for i, tid in enumerate(remaining, start=1):
        name = teams.get(tid, "?")
        print(f"  [{i:>4}/{len(remaining)}] {tid:>6}  {name[:32]:<32}", end="")

        raw, series = fetch_team(session, tid)
        if raw is None:
            failures.append(tid)
            continue

        (RAWDIR / f"{tid}.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print(f"  {len(series):>5} punten")

    # Alles samenvoegen tot een platte reeks.
    combined = OUTDIR / "ratings_history.jsonl"
    n_rows, n_teams = 0, 0
    with combined.open("w", encoding="utf-8") as fh:
        for path in sorted(RAWDIR.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            tid = path.stem
            rows = []
            for row in extract_series(raw):
                d = parse_eci_date(row.get("Date"))
                p = to_float(row.get("Points"))
                if d is None or p is None:
                    continue
                rows.append({
                    "eci_team_id": tid,
                    "team_name": teams.get(tid),
                    "date": d.isoformat(),
                    "points": p,
                    "ranking": to_float(row.get("Ranking")),
                })
            if rows:
                n_teams += 1
            for row in sorted(rows, key=lambda r: r["date"]):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"\n[OK] {n_rows:,} ratingpunten over {n_teams:,} teams")
    print(f"[OK] {combined}")
    if failures:
        print(f"[--] mislukt: {failures}")
        print("     draai het commando opnieuw; die worden dan opnieuw geprobeerd")

    print("\nInlezen met:")
    print("  df = pd.read_json('eci_history/ratings_history.jsonl', lines=True)")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ECI ratinghistorie per team")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="vergelijk historie met de match-odds feed")
    p_val.add_argument("--teams", type=int, default=5,
                       help="aantal teams om te steekproeven (default 5)")

    p_col = sub.add_parser("collect", help="haal de historie van alle teams op")
    p_col.add_argument("--limit", type=int, help="alleen de eerste N teams")
    p_col.add_argument("--source", choices=["ranking", "feed"], default="ranking",
                       help="teamlijst uit get-ranking (~1291, default) "
                            "of uit de wedstrijdfeed (~734)")

    args = parser.parse_args()

    if args.command == "validate":
        validate(args.teams)
    else:
        collect(args.limit, args.source)


if __name__ == "__main__":
    main()