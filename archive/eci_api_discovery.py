"""
eci_api_discovery.py
====================

Vraagt de WordPress REST-index van euroclubindex.com op en toont alle
geregistreerde routes. Dat is een uitputtend antwoord op de vraag "zit er
nog meer in", zonder de site te crawlen.

WordPress publiceert zelf welke endpoints bestaan:

    /wp-json/                        alle namespaces en routes
    /wp-json/happyhorizon/v1/        de routes van hun eigen thema
    /wp-json/wp/v2/                  de standaard WordPress-routes

Daar zitten ook endpoints bij waar geen enkele pagina naar linkt en die je
met crawlen dus nooit zou vinden.

Het script doet een handvol requests met pauze ertussen en schrijft alles
weg. Het haalt geen inhoud op, alleen de routebeschrijvingen.

Gebruik
-------
    python eci_api_discovery.py
    python eci_api_discovery.py --teams        (ook het teams-posttype tellen)
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
OUTDIR = Path(__file__).resolve().parent / "eci_api_discovery"
DELAY = 1.5

# Endpoints die je al kent, zodat nieuwe eruit springen.
KNOWN = {
    "/happyhorizon/v1/get-module-match-odds",
    "/happyhorizon/v1/get-module-team-details",
    "/happyhorizon/v1/get-module-latest-ranking",
    "/happyhorizon/v1/get-module-league-odds-details",
    "/happyhorizon/v1/get-module-teams-comparison",
    "/happyhorizon/v1/manifest",
}


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
        print(f"    [--] {exc}")
        return None


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def normalise(route: str) -> str:
    """'/happyhorizon/v1/get-module-match-odds' zonder trailing slash."""
    return route.rstrip("/")


def show_routes(routes: dict, namespace_filter: str | None = None) -> list[str]:
    """Print routes met hun methoden en verwachte parameters."""
    found = []

    for route, spec in sorted(routes.items()):
        if namespace_filter and namespace_filter not in route:
            continue
        # Routes met regex-placeholders zijn detailroutes; die tonen we ook,
        # maar ze zijn zonder ID niet aan te roepen.
        endpoints = spec.get("endpoints") or []
        methods = sorted({m for ep in endpoints for m in ep.get("methods", [])})

        params = set()
        for ep in endpoints:
            params.update((ep.get("args") or {}).keys())

        norm = normalise(route)
        is_new = norm not in KNOWN and not norm.endswith("/wp-json")
        marker = "NIEUW" if is_new else "     "

        print(f"  [{marker}] {route}")
        if methods:
            print(f"           methoden: {', '.join(methods)}")
        if params:
            shown = sorted(params)[:12]
            extra = f" (+{len(params) - 12})" if len(params) > 12 else ""
            print(f"           parameters: {', '.join(shown)}{extra}")

        if is_new:
            found.append(route)

    return found


def probe_index(session) -> dict:
    hr("STAP 1 - de WordPress REST-index")

    resp = get(session, f"{BASE}/wp-json/")
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "geen antwoord"
        print(f"  [--] niet op te halen ({code})")
        print("       De REST-index kan afgeschermd zijn. Dan blijft alleen")
        print("       over wat je uit de paginabronnen haalt.")
        return {}

    try:
        data = resp.json()
    except ValueError:
        print("  [--] geen JSON terug")
        return {}

    (OUTDIR / "wp-json-index.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    namespaces = data.get("namespaces") or []
    routes = data.get("routes") or {}
    print(f"  [OK] {len(namespaces)} namespaces, {len(routes)} routes")
    print(f"       namespaces: {', '.join(namespaces)}")

    return routes


def probe_theme_namespace(session) -> None:
    hr("STAP 2 - de eigen namespace van het thema (happyhorizon/v1)")

    resp = get(session, f"{BASE}/wp-json/happyhorizon/v1")
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "geen antwoord"
        print(f"  [--] niet op te halen ({code})")
        return

    try:
        data = resp.json()
    except ValueError:
        print("  [--] geen JSON terug")
        return

    (OUTDIR / "happyhorizon-v1.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    routes = data.get("routes") or {}
    print(f"  {len(routes)} routes in deze namespace:\n")
    new = show_routes(routes)

    if new:
        print(f"\n  --> {len(new)} route(s) die je nog niet gebruikte:")
        for route in new:
            print(f"      {BASE}/wp-json{route}")
    else:
        print("\n  --> geen onbekende routes; je hebt ze allemaal al gevonden")


def probe_post_types(session) -> None:
    hr("STAP 3 - posttypes (teams, competities)")

    resp = get(session, f"{BASE}/wp-json/wp/v2/types")
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "geen antwoord"
        print(f"  [--] niet op te halen ({code})")
        return

    try:
        data = resp.json()
    except ValueError:
        print("  [--] geen JSON terug")
        return

    (OUTDIR / "post-types.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    for slug, spec in sorted(data.items()):
        name = spec.get("name", "?")
        rest_base = spec.get("rest_base")
        print(f"  {slug:<20} {name:<28} rest_base={rest_base}")


def count_teams(session) -> None:
    hr("STAP 4 - hoeveel teams kent de site?")

    # per_page=1 volstaat: het aantal staat in de X-WP-Total header.
    resp = get(session, f"{BASE}/wp-json/wp/v2/teams_pt", params={"per_page": 1})
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "geen antwoord"
        print(f"  [--] niet op te halen ({code})")
        return

    total = resp.headers.get("X-WP-Total")
    pages = resp.headers.get("X-WP-TotalPages")
    print(f"  X-WP-Total: {total}   X-WP-TotalPages: {pages}")

    if total:
        print(f"\n  Je haalde er 734 uit de wedstrijdfeed.")
        try:
            diff = int(total) - 734
            if diff > 0:
                print(f"  Dat zijn er {diff} minder dan de site kent.")
                print("  Het verschil zijn clubs die deze weken niet spelen.")
            elif diff < 0:
                print("  De feed bevat er meer; mogelijk staan niet alle teams")
                print("  als los posttype geregistreerd.")
            else:
                print("  Precies evenveel; je mist niets.")
        except ValueError:
            pass

    try:
        sample = resp.json()
    except ValueError:
        return

    if isinstance(sample, list) and sample:
        keys = sorted(sample[0].keys())
        print(f"\n  velden per team: {', '.join(keys[:15])}")
        (OUTDIR / "teams_pt-sample.json").write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ontdek de REST-routes van ECI")
    parser.add_argument("--teams", action="store_true",
                        help="ook het teams-posttype tellen")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    print(f"Output: {OUTDIR}")

    session = make_session()

    routes = probe_index(session)
    if routes:
        # Alleen de thema-routes uit de index tonen; de wp/v2-routes zijn
        # standaard WordPress en niet interessant.
        theme = {k: v for k, v in routes.items() if "happyhorizon" in k}
        if theme:
            print(f"\n  thema-routes in de index: {len(theme)}\n")
            show_routes(theme)

    probe_theme_namespace(session)
    probe_post_types(session)

    if args.teams:
        count_teams(session)

    hr("KLAAR")
    print("Alle antwoorden staan als JSON in de output-map.")
    print("Een route met [NIEUW] is er een die je nog niet gebruikt.")


if __name__ == "__main__":
    main()