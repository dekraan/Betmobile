"""
eci_history_probe.py
====================

Twee vragen:

  history  Zijn er wedstrijden met ECI-kansen van vóór augustus 2026?
           Tast parameters af op get-module-match-odds en verkent de vier
           onbekende routes (get-items, get-ranking, get-competitions,
           get-module-league-odds).

  teams    Haalt de volledige teamlijst op via het WordPress-posttype
           teams_pt. De site kent er 1390; uit de wedstrijdfeed kwamen er
           734.

Aanpak: de basisaanroep geeft een datumbereik. Elke parameter die dat
bereik verder terug duwt, is een treffer. Alles wat niets doet, geeft
exact hetzelfde bereik terug.

Gebruik
-------
    python eci_history_probe.py history
    python eci_history_probe.py teams
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests ontbreekt. Installeer met: pip install requests")


BASE = "https://www.euroclubindex.com"
WPJSON = f"{BASE}/wp-json"
HH = f"{WPJSON}/happyhorizon/v1"

OUTDIR = Path(__file__).resolve().parent / "eci_history_probe"
DELAY = 1.5

# Eredivisie: overzichtelijk, herkenbare namen, altijd gevuld.
TEST_LEAGUE = "2"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Referer": f"{BASE}/",
    })
    return s


def get(session, url, **kw):
    time.sleep(DELAY)
    try:
        return session.get(url, timeout=30, **kw)
    except requests.RequestException as exc:
        print(f"      [--] {exc}")
        return None


def get_json(session, url, params=None):
    resp = get(session, url, params=params)
    if resp is None or resp.status_code != 200:
        return None, (resp.status_code if resp else None)
    try:
        return resp.json(), 200
    except ValueError:
        return None, "geen json"


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def save(name: str, data) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    (OUTDIR / f"{name}.json").write_text(text[:3_000_000], encoding="utf-8")


# ---------------------------------------------------------------------------
# Wedstrijden herkennen
# ---------------------------------------------------------------------------

def match_summary(data) -> tuple[int, str | None, str | None]:
    """Geeft (aantal wedstrijden, vroegste datum, laatste datum)."""
    rows = []

    def walk(node, depth=0):
        if depth > 8:
            return
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict) and "d_Date" in item:
                    rows.append(item)
                else:
                    walk(item, depth + 1)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)

    walk(data)
    if not rows:
        return 0, None, None

    dates = sorted(str(r.get("d_Date", ""))[:10] for r in rows if r.get("d_Date"))
    return len(rows), (dates[0] if dates else None), (dates[-1] if dates else None)


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

def probe_match_odds_params(session) -> None:
    hr("DEEL 1 - parameters op get-module-match-odds")

    url = f"{HH}/get-module-match-odds"
    base_params = {"selected_league": TEST_LEAGUE}

    data, status = get_json(session, url, base_params)
    if data is None:
        print(f"  [--] basisaanroep mislukt ({status})")
        return

    n0, first0, last0 = match_summary(data)
    print(f"  basis: {n0} wedstrijden, {first0} tot {last0}")
    print(f"  toplevel-sleutels: {list(data)[:10]}\n")
    save("baseline_match_odds", data)

    # daysBack kwam terug in de respons van team-details, dus die staat
    # vooraan. De rest zijn gebruikelijke WordPress- en API-namen.
    candidates = [
        {"days_back": 7300},
        {"daysBack": 7300},
        {"season": "2024"},
        {"season": "2024-2025"},
        {"year": "2024"},
        {"date_from": "2007-01-01"},
        {"from": "2007-01-01"},
        {"start_date": "2007-01-01"},
        {"all": "true"},
        {"show_all": "true"},
        {"history": "true"},
        {"archive": "true"},
        {"items_per_page": -1},
        {"per_page": 1000},
        {"limit": 10000},
        {"page": 2},
        {"offset": 45},
    ]

    hits = []
    for extra in candidates:
        params = {**base_params, **extra}
        label = ", ".join(f"{k}={v}" for k, v in extra.items())

        data, status = get_json(session, url, params)
        if data is None:
            print(f"  {label:<28} [--] {status}")
            continue

        n, first, last = match_summary(data)

        if first == first0 and n == n0:
            print(f"  {label:<28} geen effect")
            continue

        if first and first0 and first < first0:
            print(f"  {label:<28} *** {n} wedstrijden, terug tot {first}")
            hits.append((label, n, first, last))
            save(f"hit_{label.replace('=', '_').replace(', ', '_')}", data)
        else:
            print(f"  {label:<28} ander resultaat: {n} wedstrijden, "
                  f"{first} tot {last}")

    print()
    if hits:
        print("  TREFFER. Deze parameters geven oudere wedstrijden:")
        for label, n, first, last in hits:
            print(f"    {label}: {n} wedstrijden vanaf {first}")
    else:
        print("  Geen enkele parameter duwt het bereik terug.")
        print("  De wedstrijdfeed lijkt beperkt tot het lopende seizoen.")


def probe_unknown_routes(session) -> None:
    hr("DEEL 2 - de vier onbekende routes")

    routes = [
        ("get-items", [{}, {"selected_league": TEST_LEAGUE}, {"limit": 10}]),
        ("get-ranking", [{}, {"limit": 10}]),
        ("get-competitions", [{}]),
        ("get-module-league-odds", [{}, {"selected_league": TEST_LEAGUE}]),
    ]

    for name, param_sets in routes:
        print(f"\n  --- {name} ---")
        for params in param_sets:
            label = ", ".join(f"{k}={v}" for k, v in params.items()) or "geen parameters"

            data, status = get_json(session, f"{HH}/{name}", params)
            if data is None:
                print(f"    {label:<28} [--] {status}")
                continue

            save(f"route_{name}_{label.replace('=', '_').replace(', ', '_')}", data)

            n, first, last = match_summary(data)
            if n:
                print(f"    {label:<28} {n} wedstrijden, {first} tot {last}")
                continue

            if isinstance(data, dict):
                keys = list(data)
                print(f"    {label:<28} dict met sleutels: {keys[:10]}")
                for key in keys[:6]:
                    value = data[key]
                    if isinstance(value, list):
                        print(f"        {key}: lijst van {len(value)}")
                        if value and isinstance(value[0], dict):
                            print(f"          velden: {sorted(value[0])[:12]}")
                    elif isinstance(value, dict):
                        print(f"        {key}: dict, sleutels {list(value)[:8]}")
            elif isinstance(data, list):
                print(f"    {label:<28} lijst van {len(data)}")
                if data and isinstance(data[0], dict):
                    print(f"        velden: {sorted(data[0])[:12]}")
            else:
                print(f"    {label:<28} {type(data).__name__}: {str(data)[:80]}")


def run_history(session) -> None:
    probe_match_odds_params(session)
    probe_unknown_routes(session)

    hr("KLAAR")
    print("Alle responses staan als JSON in de output-map.")
    print("Zonder oudere wedstrijden blijft de ratinghistorie je enige")
    print("historische bron, en die gaat wel terug tot 2007.")


# ---------------------------------------------------------------------------
# teams
# ---------------------------------------------------------------------------

def run_teams(session) -> None:
    hr("Volledige teamlijst via wp/v2/teams_pt")

    url = f"{WPJSON}/wp/v2/teams_pt"
    per_page = 100
    all_rows = []
    page = 1

    while True:
        resp = get(session, url, params={
            "per_page": per_page,
            "page": page,
            "_fields": "id,slug,title,acf,link",
        })
        if resp is None:
            break
        if resp.status_code != 200:
            if page > 1:
                print(f"  pagina {page}: status {resp.status_code}, gestopt")
            else:
                print(f"  [--] status {resp.status_code}")
            break

        try:
            rows = resp.json()
        except ValueError:
            print("  [--] geen JSON")
            break

        if not rows:
            break

        all_rows.extend(rows)
        total = resp.headers.get("X-WP-Total", "?")
        print(f"  pagina {page:>3}: {len(rows):>3} teams  (totaal {len(all_rows)}/{total})")

        if len(rows) < per_page:
            break
        page += 1

    if not all_rows:
        print("  Niets opgehaald.")
        return

    save("teams_pt_full", all_rows)

    # Het ECI-team-ID zit vermoedelijk in het acf-blok.
    sample = all_rows[0]
    print(f"\n  velden: {sorted(sample)}")
    acf = sample.get("acf")
    if isinstance(acf, dict):
        print(f"  acf-velden: {sorted(acf)[:15]}")

    # Platte lijst wegschrijven met wat we kunnen herkennen.
    flat = []
    for row in all_rows:
        acf = row.get("acf") if isinstance(row.get("acf"), dict) else {}
        title = row.get("title")
        if isinstance(title, dict):
            title = title.get("rendered")
        flat.append({
            "wp_id": row.get("id"),
            "slug": row.get("slug"),
            "title": title,
            "link": row.get("link"),
            **{k: v for k, v in acf.items()
               if isinstance(v, (str, int, float)) and "id" in k.lower()},
        })

    path = OUTDIR / "teams_flat.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in flat:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  [OK] {len(flat)} teams -> {path}")
    print("  Bekijk teams_pt_full.json om te zien waar het ECI-team-ID staat.")
    print("  Daarmee kun je de historie ophalen voor teams die nu niet spelen.")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Zoek historische wedstrijden en teams")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("history", help="zoek wedstrijden van voor augustus 2026")
    sub.add_parser("teams", help="haal de volledige teamlijst op")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    print(f"Output: {OUTDIR}")

    session = make_session()
    if args.command == "history":
        run_history(session)
    else:
        run_teams(session)


if __name__ == "__main__":
    main()