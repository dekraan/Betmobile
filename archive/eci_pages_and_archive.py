"""
eci_pages_and_archive.py
========================

Twee losse controles.

  pages    Lijst alle WordPress-pagina's en competitie-posts van ECI.
           Eén call per posttype, geen crawler nodig.

  archive  Doorzoekt de Wayback Machine op gearchiveerde API-responses.
           De gearchiveerde HTML-pagina's zijn Vue- en Angular-shells
           zonder data, maar de REST-URL's staan als attribuut in die HTML
           en worden soms zelf ook gecrawld. Als daar een oude
           get-module-match-odds-respons tussen zit, is dat historische
           wedstrijddata met ECI-kansen.

Gebruik
-------
    python eci_pages_and_archive.py pages
    python eci_pages_and_archive.py archive
    python eci_pages_and_archive.py archive --fetch     (haal treffers op)
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
WPJSON = f"{BASE}/wp-json"
CDX = "https://web.archive.org/cdx/search/cdx"

OUTDIR = Path(__file__).resolve().parent / "eci_pages_archive"
DELAY = 1.5

# Posttypes uit de eerdere probe.
POST_TYPES = ["pages", "posts", "teams_pt", "leagues_pt", "options_pt"]


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
    })
    return s


def get(session, url, **kw):
    time.sleep(DELAY)
    try:
        return session.get(url, timeout=45, **kw)
    except requests.RequestException as exc:
        print(f"    [--] {exc}")
        return None


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

def list_post_type(session, rest_base: str) -> None:
    print(f"\n  --- {rest_base} ---")

    resp = get(session, f"{WPJSON}/wp/v2/{rest_base}", params={
        "per_page": 100,
        "_fields": "id,slug,link,title,date,modified",
    })
    if resp is None:
        return
    if resp.status_code != 200:
        print(f"    [--] status {resp.status_code}")
        return

    total = resp.headers.get("X-WP-Total", "?")
    pages = resp.headers.get("X-WP-TotalPages", "?")
    print(f"    totaal: {total}   pagina's: {pages}")

    try:
        rows = resp.json()
    except ValueError:
        print("    [--] geen JSON")
        return

    (OUTDIR / f"{rest_base}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    for row in rows[:60]:
        title = row.get("title")
        if isinstance(title, dict):
            title = title.get("rendered")
        modified = str(row.get("modified", ""))[:10]
        print(f"    {row.get('id'):>6}  {str(title)[:44]:<44} {modified}")

    if len(rows) > 60:
        print(f"    ... nog {len(rows) - 60}, zie {rest_base}.json")


def run_pages(session) -> None:
    hr("Alle pagina's en posts")
    for rest_base in POST_TYPES:
        list_post_type(session, rest_base)

    print("\nAlles opgeslagen als JSON. Een pagina die je niet kent en die")
    print("interessant lijkt, kun je gericht bekijken in de browser.")


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------

def cdx_query(session, url_pattern: str, extra: dict | None = None):
    """Vraag de CDX-index van de Wayback Machine op."""
    params = {
        "url": url_pattern,
        "matchType": "prefix",
        "output": "json",
        "collapse": "urlkey",
        "limit": "1000",
        "fl": "timestamp,original,mimetype,statuscode,length",
    }
    if extra:
        params.update(extra)

    resp = get(session, CDX, params=params)
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp else "geen antwoord"
        print(f"    [--] CDX gaf {code}")
        return []

    try:
        rows = resp.json()
    except ValueError:
        print("    [--] geen JSON van CDX")
        return []

    if not rows:
        return []

    header, *data = rows
    return [dict(zip(header, row)) for row in data]


def run_archive(session, do_fetch: bool) -> None:
    hr("Wayback Machine: gearchiveerde API-responses")

    patterns = [
        ("REST-endpoints van het huidige thema",
         "www.euroclubindex.com/wp-json/happyhorizon/"),
        ("alle wp-json",
         "www.euroclubindex.com/wp-json/"),
        ("oude Angular-services en api-paden",
         "www.euroclubindex.com/wp-content/themes/euroclubindex/"),
    ]

    all_hits = []

    for label, pattern in patterns:
        print(f"\n  --- {label} ---")
        print(f"      {pattern}*")
        rows = cdx_query(session, pattern)

        if not rows:
            print("      niets gearchiveerd")
            continue

        print(f"      {len(rows)} unieke URL's gearchiveerd")

        # Alleen JSON is interessant; de rest is CSS en JS.
        json_rows = [r for r in rows
                     if "json" in (r.get("mimetype") or "").lower()
                     or "wp-json" in (r.get("original") or "")]

        for row in json_rows[:25]:
            ts = row.get("timestamp", "")
            date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ts
            print(f"      {date}  {row.get('statuscode'):>3}  "
                  f"{str(row.get('length', '')):>8}  "
                  f"{row.get('original', '')[:78]}")
            all_hits.append(row)

        if len(json_rows) > 25:
            print(f"      ... nog {len(json_rows) - 25}")

    if not all_hits:
        print("\n  Geen gearchiveerde API-responses gevonden.")
        print("  Dan bevat het archief alleen de lege HTML-shells en is er")
        print("  langs deze weg geen historische wedstrijddata te halen.")
        return

    (OUTDIR / "wayback_hits.json").write_text(
        json.dumps(all_hits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [OK] {len(all_hits)} treffers -> wayback_hits.json")

    if not do_fetch:
        print("  Draai met --fetch om ze op te halen en te bekijken.")
        return

    # Ophalen, maar alleen wat op wedstrijddata lijkt.
    hr("Treffers ophalen")
    # Grootste responses eerst: die bevatten de volledige ranglijst.
    interesting = [r for r in all_hits
                   if "match-odds" in (r.get("original") or "").lower()
                   or "ranking" in (r.get("original") or "").lower()]

    if not interesting:
        print("  Geen treffers die op wedstrijd- of ranglijstdata lijken.")
        interesting = all_hits[:10]

    def size(row):
        try:
            return int(row.get("length") or 0)
        except (TypeError, ValueError):
            return 0

    interesting.sort(key=size, reverse=True)

    # Dezelfde capture komt in meerdere patronen voor.
    seen = set()
    unique = []
    for row in interesting:
        key = (row.get("timestamp"), row.get("original"))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    interesting = unique

    for row in interesting[:12]:
        ts = row.get("timestamp")
        original = row.get("original")
        print(f"\n  {ts}  ({row.get('length')} bytes)")
        print(f"    {original}")

        resp = None
        for suffix in ("id_", "if_", ""):
            url = f"https://web.archive.org/web/{ts}{suffix}/{original}"
            attempt = get(session, url)
            if attempt is None:
                continue
            if attempt.status_code == 429:
                print("    [--] 429, even wachten")
                time.sleep(20)
                attempt = get(session, url)
                if attempt is None:
                    continue
            print(f"    {suffix or '(zonder achtervoegsel)':<24} status {attempt.status_code}")
            if attempt.status_code == 200:
                resp = attempt
                break

        if resp is None:
            print("    [--] geen van de varianten gaf 200")
            continue

        try:
            data = resp.json()
        except ValueError:
            snippet = resp.text[:120].replace("\n", " ")
            print(f"    [--] geen JSON ({len(resp.content)} bytes): {snippet}")
            continue

        # Windows accepteert geen ? & % : * in bestandsnamen.
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", original.split("/")[-1] or "root")
        name = f"wb_{ts}_{safe}"[:100]
        (OUTDIR / f"{name}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False)[:2_000_000],
            encoding="utf-8")

        # Kijken of er wedstrijden in zitten.
        def count_matches(node, depth=0):
            if depth > 8:
                return 0
            if isinstance(node, list):
                n = sum(1 for i in node if isinstance(i, dict) and "d_Date" in i)
                return n or sum(count_matches(i, depth + 1) for i in node)
            if isinstance(node, dict):
                return sum(count_matches(v, depth + 1) for v in node.values())
            return 0

        n = count_matches(data)
        if n:
            print(f"    *** {n} wedstrijden in deze respons")
        else:
            keys = list(data)[:8] if isinstance(data, dict) else f"lijst van {len(data)}"
            print(f"    opgeslagen, structuur: {keys}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ECI pagina's en Wayback-archief")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pages", help="lijst alle pagina's en posts")

    p_arch = sub.add_parser("archive", help="zoek gearchiveerde API-responses")
    p_arch.add_argument("--fetch", action="store_true",
                        help="haal gevonden responses op en bekijk ze")

    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    print(f"Output: {OUTDIR}")

    session = make_session()
    if args.command == "pages":
        run_pages(session)
    else:
        run_archive(session, args.fetch)


if __name__ == "__main__":
    main()