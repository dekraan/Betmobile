"""
eci_team_history_probe.py
=========================

Zoekt uit of het team-details endpoint het volledige ratingverloop teruggeeft.

De teampagina bevat:
    data-rest-route=".../wp-json/happyhorizon/v1/get-module-team-details/"
    data-team-id="4007"
en een knop "Since 2007" die fetchData() opnieuw aanroept. Er is dus een
parameter die de periode bepaalt.

Als dit werkt, heb je achttien seizoenen ratinghistorie in plaats van de
zes weken uit de match-odds feed.

Dit script schrijft niets naar de database. Het doet een handvol requests
met pauze ertussen en bewaart alles wat het binnenkrijgt.

Gebruik
-------
    python eci_team_history_probe.py
    python eci_team_history_probe.py --team-id 80      (Telstar)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests ontbreekt. Installeer met: pip install requests")


BASE = "https://www.euroclubindex.com"
REST_URL = f"{BASE}/wp-json/happyhorizon/v1/get-module-team-details/"
JS_URL = f"{BASE}/wp-content/themes/happyhorizon/modules/module-team-details/module-team-details.js"
PAGE_URL = f"{BASE}/teams/arsenal/"

OUTDIR = Path(__file__).resolve().parent / "eci_team_history_probe"
DELAY = 1.5

# Arsenal, uit data-team-id op de teampagina.
DEFAULT_TEAM_ID = "4007"


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Referer": PAGE_URL,
    })
    return s


def get(s, url, **kw):
    time.sleep(DELAY)
    try:
        return s.get(url, timeout=30, **kw)
    except requests.RequestException as exc:
        print(f"      [--] {exc}")
        return None


# --- stap 1: de JS lezen ----------------------------------------------------

def read_js(s) -> None:
    print("\n=== STAP 1: module-team-details.js ===")
    resp = get(s, JS_URL)
    if resp is None or resp.status_code != 200:
        print("    [--] niet op te halen; ga door naar stap 2")
        return

    js = resp.text
    (OUTDIR / "module-team-details.js").write_text(js, encoding="utf-8")
    print(f"    [OK] opgeslagen ({len(js)} tekens)")

    pattern = (r"(fetch\(|URLSearchParams|team_id|teamID|season|"
               r"displayThisSeason|this_season|restRoute|data\.)")
    hits = [f"      r{i}: {ln.strip()[:150]}"
            for i, ln in enumerate(js.splitlines(), 1)
            if re.search(pattern, ln, re.IGNORECASE)]
    if hits:
        print("    -- relevante regels:")
        for h in hits[:25]:
            print(h)
        if len(hits) > 25:
            print(f"      ... nog {len(hits) - 25} regels in het bestand")


# --- stap 2: het endpoint proberen -----------------------------------------

def find_series(node, depth=0):
    """Zoek naar een tijdreeks: chart.js-stijl (labels + datasets) of een
    lijst met dicts die een datum en een waarde bevatten."""
    results = []
    if depth > 8:
        return results

    if isinstance(node, dict):
        keys = {k.lower() for k in node}
        if "labels" in keys and "datasets" in keys:
            results.append(("chartjs", node))
        for v in node.values():
            results.extend(find_series(v, depth + 1))

    elif isinstance(node, list) and node:
        if isinstance(node[0], dict):
            k = {x.lower() for x in node[0]}
            has_date = any("date" in x or x in ("d", "x", "t") for x in k)
            has_val = any(x in ("points", "eci", "index", "rating", "value", "y") for x in k)
            if has_date and has_val and len(node) > 5:
                results.append(("records", node))
        for item in node:
            results.extend(find_series(item, depth + 1))

    return results


def describe_series(kind, data) -> None:
    if kind == "chartjs":
        labels = data.get("labels") or []
        datasets = data.get("datasets") or []
        print(f"      chart.js reeks: {len(labels)} punten, {len(datasets)} datasets")
        if labels:
            print(f"        bereik: {labels[0]}  tot  {labels[-1]}")
        for ds in datasets[:3]:
            values = [v for v in (ds.get("data") or []) if isinstance(v, (int, float))]
            if not values:
                continue
            dec = sum(1 for v in values if abs(v - round(v)) > 1e-9)
            print(f"        '{ds.get('label', '?')}': {len(values)} waarden, "
                  f"{dec} met decimalen, min={min(values):.2f} max={max(values):.2f}")
    else:
        print(f"      recordreeks: {len(data)} punten")
        print(f"        velden: {sorted(data[0].keys())}")
        print(f"        eerste: {json.dumps(data[0], ensure_ascii=False)[:150]}")
        print(f"        laatste: {json.dumps(data[-1], ensure_ascii=False)[:150]}")


def probe(s, team_id: str) -> None:
    print(f"\n=== STAP 2: endpoint proberen (team {team_id}) ===")

    # Namen uit de HTML en uit vergelijkbare modules op deze site.
    id_names = ["team_id", "teamID", "selected_team", "id"]
    season_flags = [
        {},
        {"display_this_season": "false"},
        {"this_season": "false"},
        {"season": "all"},
        {"displayThisSeason": "false"},
    ]

    best = None
    for id_name in id_names:
        for flag in season_flags:
            params = {id_name: team_id, **flag}
            label = "&".join(f"{k}={v}" for k, v in params.items())
            print(f"\n    -- {label}")
            resp = get(s, REST_URL, params=params)
            if resp is None:
                continue
            print(f"       status={resp.status_code}  bytes={len(resp.content)}")
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except ValueError:
                print("       [--] geen JSON")
                continue

            fname = re.sub(r"[^A-Za-z0-9_]+", "_", label)[:60]
            (OUTDIR / f"resp_{fname}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False)[:2_000_000],
                encoding="utf-8")

            series = find_series(data)
            if not series:
                print("       geen tijdreeks gevonden")
                if isinstance(data, dict):
                    print(f"       toplevel-sleutels: {list(data)[:12]}")
                continue

            for kind, payload in series[:2]:
                describe_series(kind, payload)

            size = max(
                len(p.get("labels", [])) if k == "chartjs" else len(p)
                for k, p in series
            )
            if best is None or size > best[0]:
                best = (size, label)

            # Als we al meer dan een seizoen hebben, is de rest overbodig.
            if size > 500:
                print(f"\n    [OK] lange reeks gevonden met: {label}")
                print(f"    [OK] alles opgeslagen in {OUTDIR}")
                return

    if best:
        print(f"\n    Langste reeks: {best[0]} punten via {best[1]}")
        print("    Dat lijkt nog niet de volledige historie. Kijk in de")
        print("    opgeslagen JS welke parameter de knop 'Since 2007' zet.")
    else:
        print("\n    Geen bruikbare reeks. Stuur me de opgeslagen JS.")


def main() -> None:
    p = argparse.ArgumentParser(description="Probe ECI team-details endpoint")
    p.add_argument("--team-id", default=DEFAULT_TEAM_ID,
                   help=f"ECI team-ID uit data-team-id (default {DEFAULT_TEAM_ID} = Arsenal)")
    args = p.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    print(f"Output: {OUTDIR}")

    s = session()
    read_js(s)
    probe(s, args.team_id)


if __name__ == "__main__":
    main()