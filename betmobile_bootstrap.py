# -*- coding: utf-8 -*-
# betmobile_bootstrap.py — league_id whitelist edition + events & stats (API-FOOTBALL)
# === UNIVERSAL PROJECT-ROOT FIX ===
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ==================================

import os, json, time, requests, psycopg2, re, threading
import psycopg2.extras as pgx
from datetime import datetime, timedelta, UTC, timezone
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from collections import defaultdict
from psycopg2.extras import execute_values
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========= Config =========

from betmobile_settings import DB_CONFIG, get_api_football_key

API_KEY = get_api_football_key()

BACKFILL_DEBUG = os.getenv("BACKFILL_DEBUG", "0") == "1"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Eén Session per thread voor stabiliteit en minder TLS overhead
_SESSION_LOCAL = threading.local()

def _get_session() -> requests.Session:
    s = getattr(_SESSION_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"x-apisports-key": API_KEY, "User-Agent": "betmobile/1.0"})
        _SESSION_LOCAL.session = s
    return s

SEASON = int(os.getenv("SEASON", "2025"))

# Rate
TOKENS_PER_MIN = int(os.getenv("TOKENS_PER_MIN", "445"))
SLEEP_ON_429 = os.getenv("SLEEP_ON_429", "1") == "1"
MIN_SLEEP_ON_429 = int(os.getenv("MIN_SLEEP_ON_429", "10"))
RESPECT_PROVIDER_LIMITS = os.getenv("RESPECT_PROVIDER_LIMITS", "1") == "1"
RETRY_ATTEMPTS = 5

# Fixtures window
FIXTURES_MODE = os.getenv("FIXTURES_MODE", "BY_LEAGUE").upper()  # DATE_GLOBAL | BY_LEAGUE
HORIZON_PAST_DAYS = int(os.getenv("HORIZON_PAST_DAYS", "7"))
HORIZON_FUTURE_DAYS = int(os.getenv("HORIZON_FUTURE_DAYS", "14"))

# Events/stats toggles
LOAD_EVENTS = os.getenv("LOAD_EVENTS", "1") == "1"
LOAD_STATS  = os.getenv("LOAD_STATS", "0") == "1"
EVENTS_WINDOW_DAYS = int(os.getenv("EVENTS_WINDOW_DAYS", "30"))
STATS_WINDOW_DAYS  = int(os.getenv("STATS_WINDOW_DAYS", "30"))

# In-memory de-dupe
SEEN_TEAMS = set()

# ====== RUN MODES ======
RUN_MODE = os.getenv("RUN_MODE", "FULL").upper()
CLEAN_EXCLUDED = os.getenv("CLEAN_EXCLUDED", "0") == "1"

MODE_PRESETS = {
    # steps + fixture horizon + defaults voor events/stats
    "FULL": {
        "do_step_1": True, "do_step_2": True, "do_step_3": True,
        "do_step_4": True, "do_step_5": True, "do_clean": CLEAN_EXCLUDED,
        "past_days": 7, "future_days": 14,
        "events_on": True,  "stats_on": True,
        "events_days": 30,  "stats_days": 30,
    },
    "MAINTENANCE": {
        "do_step_1": True, "do_step_2": True, "do_step_3": True,
        "do_step_4": True, "do_step_5": True, "do_clean": CLEAN_EXCLUDED,
        "past_days": 4, "future_days": 7,
        "events_on": True,  "stats_on": True,
        "events_days": 7,  "stats_days": 7,
    },
    "ODDS_ONLY": {
        "do_step_1": False,"do_step_2": False,"do_step_3": False,
        "do_step_4": True, "do_step_5": True, "do_clean": False,
        "past_days": 0, "future_days": 3,
        "events_on": False, "stats_on": False,
        "events_days": 0,   "stats_days": 0,
    },
    "HIST_BACKFILL": {
        "do_step_1": True,"do_step_2": True,"do_step_3": False,
        "do_step_4": False,"do_step_5": False,"do_clean": False,
        "past_days": 0, "future_days": 0,
        # laat events/stats hier door ENV bepalen (geen default aanpassing)
        "events_on": False, "stats_on": False,
        "events_days": 0,   "stats_days": 0,
    },
    "CLEAN": {
        "do_step_1": False,"do_step_2": False,"do_step_3": False,
        "do_step_4": False,"do_step_5": False,"do_clean": True,
        "past_days": 0, "future_days": 0,
        "events_on": False, "stats_on": False,
        "events_days": 0,   "stats_days": 0,
    },
}

def _int_env(name, default):
    try:
        v = os.getenv(name)
        return int(v) if v is not None else default
    except:
        return default

def _bool_env(name, default: bool):
    v = os.getenv(name)
    if v is None:
        return default
    return v == "1"

def get_mode_config():
    preset = MODE_PRESETS.get(RUN_MODE, MODE_PRESETS["FULL"]).copy()
    preset["past_days"]   = _int_env("HORIZON_PAST_DAYS",  preset["past_days"])
    preset["future_days"] = _int_env("HORIZON_FUTURE_DAYS", preset["future_days"])

    # aan/uit: ENV wint altijd, tenzij ENV niet gezet is
    preset["events_on"] = _bool_env("LOAD_EVENTS", preset["events_on"])
    preset["stats_on"]  = _bool_env("LOAD_STATS",  preset["stats_on"])

    # windows: env > preset > globale defaults
    preset["events_days"] = _int_env("EVENTS_WINDOW_DAYS", preset.get("events_days", EVENTS_WINDOW_DAYS))
    preset["stats_days"]  = _int_env("STATS_WINDOW_DAYS",  preset.get("stats_days",  STATS_WINDOW_DAYS))
    return preset

def _print_plan(cfg):
    print("\n== Run plan ==")
    print(f"RUN_MODE : {RUN_MODE}")
    print(
        f"Steps : "
        f"1={'Y' if cfg['do_step_1'] else '-'}, "
        f"2={'Y' if cfg['do_step_2'] else '-'}, "
        f"3={'Y' if cfg['do_step_3'] else '-'}, "
        f"4={'Y' if cfg['do_step_4'] else '-'}, "
        f"5={'Y' if cfg['do_step_5'] else '-'}, "
        f"CLEAN={'Y' if cfg['do_clean'] else '-'}"
    )
    ev = f"on (last {cfg['events_days']}d)" if cfg["events_on"] else "off"
    st = f"on (last {cfg['stats_days']}d)"  if cfg["stats_on"]  else "off"
    print(f"HORIZON : past={cfg['past_days']}d future={cfg['future_days']}d")
    print(f"EVENTS : {ev}, STATS : {st}\n")
    print(
    "Overrides : "
    f"HORIZON_PAST_DAYS={HORIZON_PAST_DAYS}, "
    f"HORIZON_FUTURE_DAYS={HORIZON_FUTURE_DAYS}, "
    f"LOAD_EVENTS={int(cfg['events_on'])}, "
    f"LOAD_STATS={int(cfg['stats_on'])}\n"
    )

# ========= Infra =========

def get_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as c:
        # korte timeouts voorkomen hangers; pas waarden evt. aan
        c.execute("SET statement_timeout = '30s';")
        c.execute("SET lock_timeout = '5s';")
    return conn

def get_season_for_league(league_id: int, fallback_season: int = SEASON) -> int:
    """
    Pak voor een league het beste seizoen uit tabel seasons:
    1. current = true
    2. anders hoogste season
    3. anders fallback
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT season
            FROM seasons
            WHERE league_id = %s
              AND current = true
            ORDER BY season DESC
            LIMIT 1
        """, (league_id,))
        row = cur.fetchone()
        if row:
            return int(row[0])

        cur.execute("""
            SELECT season
            FROM seasons
            WHERE league_id = %s
            ORDER BY season DESC
            LIMIT 1
        """, (league_id,))
        row = cur.fetchone()
        if row:
            return int(row[0])

    return int(fallback_season)

def ensure_schema():
    ddl = """
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE IF NOT EXISTS api_cache (
        endpoint TEXT PRIMARY KEY,
        payload JSONB NOT NULL,
        fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS leagues (
        league_id INTEGER PRIMARY KEY, name TEXT, type TEXT, country TEXT,
        country_code TEXT, logo TEXT, flag TEXT
    );

    CREATE TABLE IF NOT EXISTS seasons (
        league_id INTEGER REFERENCES leagues(league_id) ON DELETE CASCADE,
        season INTEGER, start_date DATE, end_date DATE, current BOOLEAN,
        coverage JSONB,
        PRIMARY KEY (league_id, season)
    );

    CREATE TABLE IF NOT EXISTS venues (
        venue_id INTEGER PRIMARY KEY, name TEXT, address TEXT, city TEXT,
        capacity INTEGER, surface TEXT, image TEXT
    );

    CREATE TABLE IF NOT EXISTS teams (
        team_id INTEGER PRIMARY KEY, name TEXT, code TEXT, country TEXT,
        founded INTEGER, national BOOLEAN, logo TEXT,
        venue_id INTEGER REFERENCES venues(venue_id)
    );

    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id INTEGER PRIMARY KEY,
        league_id INTEGER REFERENCES leagues(league_id),
        season INTEGER, date_utc TIMESTAMPTZ, timezone TEXT,
        status_short TEXT, status_long TEXT, referee TEXT,
        venue_id INTEGER REFERENCES venues(venue_id), round TEXT,
        home_team_id INTEGER REFERENCES teams(team_id),
        away_team_id INTEGER REFERENCES teams(team_id),
        home_goals INTEGER, away_goals INTEGER,
        winner_home BOOLEAN, winner_away BOOLEAN, winner_draw BOOLEAN,
        ht_home_goals INTEGER, ht_away_goals INTEGER,
        ft_home_goals INTEGER, ft_away_goals INTEGER,
        et_home_goals INTEGER, et_away_goals INTEGER,
        pen_home_goals INTEGER, pen_away_goals INTEGER
    );

    -- veilige ALTERs (no-op als kolommen al bestaan)
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS ht_home_goals INTEGER;
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS ht_away_goals INTEGER;
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS ft_home_goals INTEGER;
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS ft_away_goals INTEGER;
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS et_home_goals INTEGER;
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS et_away_goals INTEGER;
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS pen_home_goals INTEGER;
    ALTER TABLE fixtures ADD COLUMN IF NOT EXISTS pen_away_goals INTEGER;

    CREATE TABLE IF NOT EXISTS odds_bookmakers (
        bookmaker_id INTEGER PRIMARY KEY, name TEXT
    );

    CREATE TABLE IF NOT EXISTS odds_markets (
        market_key TEXT PRIMARY KEY, name TEXT
    );

    CREATE TABLE IF NOT EXISTS odds_values (
        fixture_id INTEGER REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
        bookmaker_id INTEGER REFERENCES odds_bookmakers(bookmaker_id),
        market_key TEXT REFERENCES odds_markets(market_key),
        label TEXT, odd NUMERIC, last_update TIMESTAMPTZ,
        PRIMARY KEY (fixture_id, bookmaker_id, market_key, label)
    );

    CREATE TABLE IF NOT EXISTS odds_values_snapshots (
        fixture_id   INTEGER NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
        bookmaker_id INTEGER NOT NULL REFERENCES odds_bookmakers(bookmaker_id),
        market_key   TEXT    NOT NULL REFERENCES odds_markets(market_key),
        label        TEXT    NOT NULL,
        odd          NUMERIC NOT NULL,
        captured_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (fixture_id, bookmaker_id, market_key, label, captured_at)
    );
    CREATE INDEX IF NOT EXISTS idx_snapshots_fx ON odds_values_snapshots (fixture_id, market_key, label, captured_at DESC);
    CREATE INDEX IF NOT EXISTS idx_snapshots_captured ON odds_values_snapshots (captured_at);

    -- seed markten (idempotent)
    INSERT INTO odds_markets(market_key,name) VALUES
      ('btts','Both Teams To Score'),
      ('ou_2_5','Over/Under 2.5'),
      ('dnb','Draw No Bet'),
      ('ah','Asian Handicap')
    ON CONFLICT (market_key) DO NOTHING;

    INSERT INTO odds_markets(market_key,name) VALUES
    ('1x2','Match Winner'),
    ('fh_ou_0_5','1st Half Over/Under 0.5'),
    ('fh_ou_1_5','1st Half Over/Under 1.5'),
    ('fh_1x2','First Half Match Winner'),
    ('corners_ou','Corners Over/Under')
    ON CONFLICT (market_key) DO NOTHING;

    CREATE TABLE IF NOT EXISTS league_policy (
        league_id INTEGER PRIMARY KEY, country TEXT, league_name TEXT,
        league_type TEXT, policy TEXT CHECK (policy IN ('include','exclude')),
        reason TEXT, notes TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_league_policy_policy ON league_policy(policy);

    CREATE INDEX IF NOT EXISTS idx_odds_values_fixture ON odds_values(fixture_id);
    CREATE INDEX IF NOT EXISTS idx_odds_values_market ON odds_values(market_key);
    CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date_utc);

    CREATE TABLE IF NOT EXISTS backfill_progress (
        league_id INTEGER, window_start DATE, window_end DATE,
        fixtures_tried INTEGER, odds_calls INTEGER, odds_rows INTEGER,
        done_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (league_id, window_start, window_end)
    );

    -- events
    CREATE TABLE IF NOT EXISTS fixture_events (
        fixture_id    BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
        event_time    INTEGER,
        event_extra   INTEGER,
        event_extra0  INTEGER NOT NULL DEFAULT 0,
        team_id       INTEGER,
        player_id     INTEGER,
        player_name   TEXT,
        assist_id     INTEGER,
        assist_name   TEXT,
        type          TEXT,
        detail        TEXT,
        comments      TEXT
        -- PK zetten we na generated kolommen
    );
    CREATE INDEX IF NOT EXISTS ix_fixture_events_fixture ON fixture_events(fixture_id);

    -- team stats
    CREATE TABLE IF NOT EXISTS fixture_statistics_team (
        fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
        team_id    INTEGER NOT NULL,
        stats      JSONB NOT NULL,
        PRIMARY KEY (fixture_id, team_id)
    );
    """

    with get_conn() as conn, conn.cursor() as cur:
        # 1) voer het volledige DDL-blok uit
        cur.execute(ddl)

        # 2) generated kolommen toevoegen (idempotent)
        cur.execute("""
        ALTER TABLE fixture_events
          ADD COLUMN IF NOT EXISTS team_id0   INTEGER GENERATED ALWAYS AS (COALESCE(team_id, 0)) STORED,
          ADD COLUMN IF NOT EXISTS player_id0 INTEGER GENERATED ALWAYS AS (COALESCE(player_id, 0)) STORED;
        """)

        # 3) primaire sleutel zetten indien nog niet aanwezig (gebruikt *_id0 kolommen)
        cur.execute("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'fixture_events'::regclass
              AND contype  = 'p'
          ) THEN
            ALTER TABLE fixture_events
              ADD CONSTRAINT fixture_events_pk PRIMARY KEY
              (fixture_id, event_time, event_extra0, team_id0, player_id0, type, detail);
          END IF;
        END$$;
        """)

        conn.commit()


class TokenBucket:
    def __init__(self, capacity, refill_per_second):
        self.capacity = capacity
        self.tokens = capacity
        self.refill = refill_per_second
        self.last = time.monotonic()
        self.lock = Lock()

    def consume(self, amount=1):
        with self.lock:
            now = time.monotonic()
            delta = now - self.last
            self.tokens = min(self.capacity, self.tokens + delta * self.refill)
            self.last = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

bucket = TokenBucket(capacity=TOKENS_PER_MIN, refill_per_second=TOKENS_PER_MIN/60.0)

def cache_get(key, max_age_sec):
    if max_age_sec == 0:
        return None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT payload, fetched_at FROM api_cache WHERE endpoint=%s", (key,))
        row = cur.fetchone()
        if not row:
            return None
        payload, fetched_at = row
        age = (datetime.now(UTC) - fetched_at).total_seconds()
        return payload if age <= max_age_sec else None

def cache_put(key, payload, max_age_sec):
    if max_age_sec == 0:
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO api_cache(endpoint, payload, fetched_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (endpoint) DO UPDATE
            SET payload=EXCLUDED.payload, fetched_at=EXCLUDED.fetched_at
            """,
            (key, json.dumps(payload))
        )
        conn.commit()

@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(RETRY_ATTEMPTS), retry=retry_if_exception_type((requests.exceptions.RequestException,)))
def _get(path, params=None):
    while True:
        while not bucket.consume():
            time.sleep(0.25)
        # mini-jitter per call
        time.sleep(0.05)

        r = _get_session().get(f"{BASE_URL}{path}", params=params, timeout=30)

        h = {k.lower(): v for k, v in r.headers.items()}
        def _i(*keys, default=0):
            for k in keys:
                if k in h and str(h[k]).strip() != "":
                    try: return max(0, int(h[k]))
                    except: pass
            return default

        min_limit = _i("x-ratelimit-requests-limit", "x-ratelimit-minute-limit")
        min_rem   = _i("x-ratelimit-requests-remaining", "x-ratelimit-minute-remaining")
        min_reset = _i("x-ratelimit-requests-reset", "x-ratelimit-minute-reset")
        day_rem   = _i("x-ratelimit-day-remaining")
        day_reset = _i("x-ratelimit-day-reset")
        mon_rem   = _i("x-ratelimit-month-remaining")
        mon_reset = _i("x-ratelimit-month-reset")

        if r.status_code == 429:
            if not SLEEP_ON_429:
                raise requests.exceptions.RequestException(f"429 Too Many Requests: {r.text}")
            now_epoch = int(time.time())  # headers zijn epoch-based
            candidates = [t for t in (min_reset, day_reset, mon_reset) if t and t > now_epoch]
            sleep_s = max(MIN_SLEEP_ON_429, min(candidates) - now_epoch) if candidates else MIN_SLEEP_ON_429
            print(f"[rate] 429 ontvangen. Wacht {sleep_s}s (min_reset={min_reset}, day_reset={day_reset}, month_reset={mon_reset}).")
            time.sleep(sleep_s)
            continue

        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("errors"):
            print(f"[api][errors-inline] {data.get('errors')}")

        if RESPECT_PROVIDER_LIMITS:
            wait_s, now = None, int(time.time())
            if min_limit and min_rem == 0:
                wait_s = max(1, (min_reset - now)) if (min_reset and min_reset > now) else MIN_SLEEP_ON_429
            if day_rem == 0 and day_reset and (wait_s is None or day_reset - now > wait_s):
                wait_s = max(MIN_SLEEP_ON_429, day_reset - now)
            if mon_rem == 0 and mon_reset and (wait_s is None or mon_reset - now > wait_s):
                wait_s = max(MIN_SLEEP_ON_429, mon_reset - now)
            if wait_s:
                print(f"[rate] Remaining=0. Wacht {wait_s}s tot reset (min_rem={min_rem}, day_rem={day_rem}, month_rem={mon_rem}).")
                time.sleep(wait_s)

        return data

def get_json(path, params=None, cache_ttl=3600):
    key = path + "|" + json.dumps(params or {}, sort_keys=True)
    cached = cache_get(key, cache_ttl)
    if cached is not None:
        return cached
    data = _get(path, params=params)
    cache_put(key, data, cache_ttl)
    return data

def get_all_pages(path, params=None, cache_ttl=900):
    base = dict(params or {})
    data = get_json(path, params=base, cache_ttl=cache_ttl)
    all_resp = (data or {}).get("response") or []
    paging = (data or {}).get("paging") or {}
    try:
        total = int(paging.get("total") or 1)
    except Exception:
        total = 1
    for page in range(2, total + 1):
        p = dict(base); p["page"] = page
        data = get_json(path, params=p, cache_ttl=cache_ttl)
        all_resp.extend((data or {}).get("response") or [])
    return all_resp

def bulk_upsert(table, cols, rows, conflict_cols):
    if not rows:
        return
    idx_map = {c: cols.index(c) for c in conflict_cols}
    uniq = {}
    for r in rows:
        key = tuple(r[idx_map[c]] for c in conflict_cols)
        uniq[key] = r
    deduped_rows = list(uniq.values())

    cols_list = ", ".join(cols)
    placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
    updates = ", ".join([f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict_cols])

    sql = f"""
    INSERT INTO {table} ({cols_list})
    VALUES %s
    ON CONFLICT ({", ".join(conflict_cols)})
    DO UPDATE SET {updates}
    """
    with get_conn() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, deduped_rows, template=placeholders, page_size=500)
        conn.commit()

# ========= Stap 1: leagues + seasons =========

def load_leagues_and_seasons(base_season=SEASON):
    """
    Haal leagues/seasons op voor meerdere seasons rond de opgegeven base_season.
    Zo missen we geen leagues die inmiddels in een nieuw kalenderjaar-seizoen zitten.
    """
    seasons_to_pull = sorted({base_season - 1, base_season, base_season + 1})

    all_leagues = {}
    all_seasons = {}

    for season in seasons_to_pull:
        data = get_json("/leagues", params={"season": season}, cache_ttl=24*3600)
        resp = data.get("response", [])

        for item in resp:
            league = item.get("league") or {}
            country = item.get("country") or {}
            seasons = item.get("seasons") or []

            league_id = league.get("id")
            if not league_id:
                continue

            all_leagues[league_id] = (
                league_id,
                league.get("name"),
                league.get("type"),
                country.get("name"),
                country.get("code"),
                league.get("logo"),
                country.get("flag"),
            )

            for s in seasons:
                year = s.get("year")
                if year is None:
                    continue

                key = (league_id, year)
                all_seasons[key] = (
                    league_id,
                    year,
                    (s.get("start") or "")[:10] or None,
                    (s.get("end") or "")[:10] or None,
                    s.get("current"),
                    json.dumps(s.get("coverage") or {})
                )

    leagues_rows = list(all_leagues.values())
    seasons_rows = list(all_seasons.values())

    bulk_upsert(
        "leagues",
        ["league_id","name","type","country","country_code","logo","flag"],
        leagues_rows,
        ["league_id"]
    )
    bulk_upsert(
        "seasons",
        ["league_id","season","start_date","end_date","current","coverage"],
        seasons_rows,
        ["league_id","season"]
    )

    print(f"[leagues] pulled_seasons={seasons_to_pull} upserted_leagues={len(leagues_rows)} upserted_seasons={len(seasons_rows)}")

# ========= Stap 2: whitelist policy =========

REQUIRE_ODDS_COVER = os.getenv("REQUIRE_ODDS_COVERAGE", "0") == "1"

def league_ids_from_aliases():
    """
    Haal alle actieve api_league_id's op uit league_aliases.
    Dit is vanaf nu de enige bron voor welke leagues we scrapen.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT api_league_id
            FROM league_aliases
            WHERE is_active = true
              AND api_league_id IS NOT NULL
            ORDER BY api_league_id
        """)
        rows = cur.fetchall()

    return [int(r[0]) for r in rows]


def print_league_ids_from_aliases():
    ids = league_ids_from_aliases()
    print("\n=== League IDs from league_aliases ===")
    print(f"count={len(ids)}")
    row = []
    for i, lid in enumerate(ids, 1):
        row.append(str(lid))
        if i % 12 == 0:
            print(" " + ", ".join(row))
            row = []
    if row:
        print(" " + ", ".join(row))
    print("=== /League IDs ===\n")
    return ids

def build_league_policy_include_only(season_for_policy: int):
    alias_ids = print_league_ids_from_aliases()

    if not alias_ids:
        print("[policy] WARNING: geen actieve api_league_id's gevonden in league_aliases.")
        return

    # Eerst alles in leagues baseline op exclude zetten
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO league_policy(league_id, country, league_name, league_type, policy, reason)
            SELECT l.league_id,
                   COALESCE(l.country, ''),
                   COALESCE(l.name, ''),
                   COALESCE(l.type, ''),
                   'exclude',
                   'not_in_league_aliases'
            FROM leagues l
            ON CONFLICT (league_id) DO UPDATE
            SET country     = EXCLUDED.country,
                league_name = EXCLUDED.league_name,
                league_type = EXCLUDED.league_type,
                policy      = 'exclude',
                reason      = 'not_in_league_aliases',
                updated_at  = NOW();
        """)
        conn.commit()

    # Controle: welke alias_ids bestaan echt in leagues?
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT league_id
            FROM leagues
            WHERE league_id = ANY(%s)
        """, (alias_ids,))
        present = {r[0] for r in cur.fetchall()}

    missing = [lid for lid in alias_ids if lid not in present]
    if missing:
        print(f"[policy] WARNING: api_league_id(s) uit league_aliases niet gevonden in leagues: {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    include_ids = [lid for lid in alias_ids if lid in present]

    # Zet alleen leagues uit league_aliases op include
    updated = 0
    if include_ids:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE league_policy
                SET policy='include',
                    reason='league_aliases',
                    updated_at=NOW()
                WHERE league_id = ANY(%s)
            """, (include_ids,))
            updated = cur.rowcount
            conn.commit()

    if REQUIRE_ODDS_COVER:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE league_policy lp
                SET policy='exclude',
                    reason='no_odds_coverage_in_season',
                    updated_at=NOW()
                WHERE policy='include'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM seasons s
                    WHERE s.league_id = lp.league_id
                      AND s.season = %s
                      AND COALESCE((s.coverage->>'odds')::boolean, false) = true
                );
            """, (season_for_policy,))
            conn.commit()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM league_policy")
        tot = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM league_policy WHERE policy='include'")
        inc = cur.fetchone()[0]

    print("\n=== League policy (from league_aliases) ===")
    print(f"include={inc}/{tot} | updated={updated}")
    print("reason='league_aliases' voor leagues die we willen scrapen")
    print("=== /policy ===\n")

# ========= CLEANUP =========

def cleanup_excluded_data():
    print("== CLEANUP: remove excluded leagues' data + orphan teams/venues ==")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM fixtures"); f_before = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM odds_values"); o_before = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM teams"); t_before = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM venues"); v_before = cur.fetchone()[0]

        cur.execute("""
            DELETE FROM fixtures f USING league_policy lp
            WHERE lp.league_id=f.league_id AND lp.policy='exclude';
        """); f_del = cur.rowcount

        cur.execute("""
            DELETE FROM teams t WHERE NOT EXISTS (
                SELECT 1 FROM fixtures f WHERE f.home_team_id=t.team_id OR f.away_team_id=t.team_id
            );
        """); t_del = cur.rowcount

        cur.execute("""
            DELETE FROM venues v WHERE NOT EXISTS (SELECT 1 FROM teams t WHERE t.venue_id=v.venue_id)
              AND NOT EXISTS (SELECT 1 FROM fixtures f WHERE f.venue_id=v.venue_id);
        """); v_del = cur.rowcount

        conn.commit()

        cur.execute("SELECT COUNT(*) FROM fixtures"); f_after = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM odds_values"); o_after = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM teams"); t_after = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM venues"); v_after = cur.fetchone()[0]

    print(f"[clean] fixtures: {f_before} -> {f_after} (deleted {f_del})")
    print(f"[clean] odds_values: {o_before} -> {o_after} (via cascade)")
    print(f"[clean] teams: {t_before} -> {t_after} (deleted {t_del})")
    print(f"[clean] venues: {v_before} -> {v_after} (deleted {v_del})")

# ========= Stap 3: teams & venues =========

def load_teams_for_policy(season=SEASON):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT lp.league_id, l.country, l.name
            FROM league_policy lp
            JOIN leagues l ON l.league_id = lp.league_id
            WHERE lp.policy='include'
              AND EXISTS (
                  SELECT 1
                  FROM seasons s
                  WHERE s.league_id = lp.league_id
              )
            ORDER BY l.country, l.name
        """)
        leagues = cur.fetchall()

    teams_rows, venues_rows = [], []
    total = len(leagues)
    print(f"[teams] allowed_leagues={total} (season={season})")
    for idx, (lg, country, name) in enumerate(leagues, start=1):
        league_season = get_season_for_league(lg, fallback_season=season)
        tag = f"{idx}/{total} league={lg} {country} - {name} season={league_season}"
        try:
            data = get_json("/teams", params={"league": lg, "season": league_season}, cache_ttl=24*3600)
            resp = data.get("response", [])
            if idx == 1 or idx % 5 == 0:
                print(f"[teams] {tag} teams={len(resp)}")
            for item in resp:
                team = item.get("team") or {}
                venue = item.get("venue") or {}
                if venue.get("id"):
                    venues_rows.append((
                        venue.get("id"), venue.get("name"), venue.get("address"),
                        venue.get("city"), venue.get("capacity"), venue.get("surface"),
                        venue.get("image")
                    ))
                teams_rows.append((
                    team.get("id"), team.get("name"), team.get("code"),
                    team.get("country"), team.get("founded"), team.get("national"),
                    team.get("logo"), venue.get("id") if venue.get("id") else None
                ))
        except Exception as e:
            print(f"[teams][WARN] {tag} error: {e}")

        if idx % 3 == 0 or idx == total:
            if venues_rows:
                bulk_upsert("venues", ["venue_id","name","address","city","capacity","surface","image"], venues_rows, ["venue_id"])
                venues_rows.clear()
            if teams_rows:
                bulk_upsert("teams", ["team_id","name","code","country","founded","national","logo","venue_id"], teams_rows, ["team_id"])
                teams_rows.clear()
    print("[teams] klaar.")

# ========= Helpers: fixtures =========

def _row_from_fixture_item(item):
    fixture = item.get("fixture") or {}
    league = item.get("league") or {}
    teams = item.get("teams") or {}
    goals = item.get("goals") or {}
    score = item.get("score") or {}

    home, away = teams.get("home") or {}, teams.get("away") or {}
    # originele winner flags van API (kunnen None zijn)
    winner_home_api = home.get("winner")
    winner_away_api = away.get("winner")

    # kies eindstand: eerst fulltime als beschikbaar, anders 'goals'
    ft_h, ft_a = (score.get("fulltime") or {}).get("home"), (score.get("fulltime") or {}).get("away")
    g_h,  g_a  = goals.get("home"), goals.get("away")
    end_h = ft_h if ft_h is not None else g_h
    end_a = ft_a if ft_a is not None else g_a

    # is de wedstrijd klaar?
    status = (fixture.get("status") or {}).get("short") or ""
    finished = status in ("FT","AET","PEN","AWD","WO")

    # default: laat None voor niet-afgeronde wedstrijden
    winner_home = winner_home_api
    winner_away = winner_away_api
    winner_draw = None

    if finished and end_h is not None and end_a is not None:
        if end_h > end_a:
            winner_home, winner_away, winner_draw = True, False, False
        elif end_a > end_h:
            winner_home, winner_away, winner_draw = False, True, False
        else:
            winner_home, winner_away, winner_draw = False, False, True
    else:
        # wedstrijd niet klaar: als API wél een winnaar gaf, forceer draw=False voor consistentie
        if winner_home_api is True:
            winner_away, winner_draw = False, False
        elif winner_away_api is True:
            winner_home, winner_draw = False, False
        elif winner_home_api is False and winner_away_api is False:
            # API zegt expliciet geen winnaar: als we een gelijke stand kennen, zet draw=True
            if end_h is not None and end_a is not None and end_h == end_a:
                winner_draw = True

    raw_vid = (fixture.get("venue") or {}).get("id")
    venue_id = raw_vid if (raw_vid and int(raw_vid) > 0) else None

    def g(path, key):
        d = (score.get(path) or {}); return d.get(key)

    return (
        fixture.get("id"), league.get("id"), league.get("season"),
        fixture.get("date"), fixture.get("timezone"),
        (fixture.get("status") or {}).get("short"),
        (fixture.get("status") or {}).get("long"),
        fixture.get("referee"), venue_id, league.get("round"),
        home.get("id"), away.get("id"),
        goals.get("home"), goals.get("away"),
        winner_home, winner_away, winner_draw,
        g("halftime","home"), g("halftime","away"),
        g("fulltime","home"), g("fulltime","away"),
        g("extratime","home"), g("extratime","away"),
        g("penalty","home"), g("penalty","away"),
    )

def _collect_fixture_venue(item):
    v = (item.get("fixture") or {}).get("venue") or {}
    vid = v.get("id")
    if not vid or int(vid) <= 0:
        return None
    return (
        vid, v.get("name"), v.get("address"), v.get("city"),
        v.get("capacity"), v.get("surface"), v.get("image"),
    )

def _ensure_teams_exist(team_ids):
    if not team_ids:
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT team_id FROM teams WHERE team_id = ANY(%s)", (list(team_ids),))
        have = {r[0] for r in cur.fetchall()}
    missing = [tid for tid in team_ids if tid and tid not in have and tid not in SEEN_TEAMS]
    if not missing:
        return
    SEEN_TEAMS.update(missing)
    teams_rows, venues_rows = [], []
    for tid in missing:
        try:
            data = get_json("/teams", params={"id": tid}, cache_ttl=48*3600)
            for item in data.get("response", []):
                team = item.get("team") or {}
                venue = item.get("venue") or {}
                if venue.get("id"):
                    venues_rows.append((
                        venue.get("id"), venue.get("name"), venue.get("address"),
                        venue.get("city"), venue.get("capacity"), venue.get("surface"),
                        venue.get("image")
                    ))
                teams_rows.append((
                    team.get("id"), team.get("name"), team.get("code"),
                    team.get("country"), team.get("founded"), team.get("national"),
                    team.get("logo"), venue.get("id") if venue.get("id") else None
                ))
        except Exception as e:
            print(f"[teams][LAZY][WARN] team_id={tid} error: {e}")
    if venues_rows:
        bulk_upsert("venues", ["venue_id","name","address","city","capacity","surface","image"], venues_rows, ["venue_id"])
    if teams_rows:
        bulk_upsert("teams", ["team_id","name","code","country","founded","national","logo","venue_id"], teams_rows, ["team_id"])

def _upsert_fixtures(rows):
    if not rows:
        return
    team_ids = set()
    for r in rows:
        if len(r) >= 12:
            team_ids.add(r[10]); team_ids.add(r[11])
    _ensure_teams_exist(team_ids)
    bulk_upsert("fixtures", [
        "fixture_id","league_id","season","date_utc","timezone",
        "status_short","status_long","referee","venue_id","round",
        "home_team_id","away_team_id","home_goals","away_goals",
        "winner_home","winner_away","winner_draw",
        "ht_home_goals","ht_away_goals","ft_home_goals","ft_away_goals",
        "et_home_goals","et_away_goals","pen_home_goals","pen_away_goals"
    ], rows, ["fixture_id"])

def _date_list(past_days, future_days):
    today = datetime.now(UTC).date()
    return [(today + timedelta(days=d)).isoformat() for d in range(-past_days, future_days+1)]

def _norm_label(x):
    """Zet label/value veilig om naar een genormaliseerde string."""
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return str(x)
    try:
        return str(x).strip().lower()
    except Exception:
        return repr(x).strip().lower()

# ========= Odds scope =========

ALLOWED_BOOKMAKERS = {8, 4}   # 8=Bet365, 4=Pinnacle

BET_ID_1X2 = 1                # Match Winner
BET_ID_AH = 4                 # Asian Handicap
BET_ID_OU = 5                 # Goals Over/Under

# ========= Odds helpers =========

def _expected_labels(market_key: str) -> set[str] | None:
    # None = geen gating
    if market_key == "btts":
        return {"yes", "no"}
    if market_key == "ou_2_5":
        return {"over_2.5", "under_2.5"}
    if market_key == "1x2":
        return {"Home", "Draw", "Away"}
    if market_key == "dnb":
        return {"home", "away"}
    if market_key == "fh_ou_0_5":
        return {"over_0.5", "under_0.5"}
    if market_key == "fh_ou_1_5":
        return {"over_1.5", "under_1.5"}
    if market_key == "fh_1x2":
        return {"Home", "Draw", "Away"}
    return None


def _buf_add(buf, fixture_id, bookmaker_id, market_key, label, odd, last_upd):
    if bookmaker_id is None or odd is None or not label:
        return
    key = (fixture_id, bookmaker_id, market_key)
    d = buf.setdefault(key, {"last_upd": last_upd, "vals": {}})
    # “laatste wint”
    d["vals"][label] = odd
    if last_upd:
        d["last_upd"] = last_upd


def _flush_buf_to_rows(buf, out_rows):
    """
    Zet buffer om naar out_rows (fixture_id, bookmaker_id, market_key, label, odd, last_update)
    Alleen complete sets voor markten met expected_labels.
    """
    for (fx, bmid, mkey), payload in buf.items():
        vals = payload["vals"]
        last_upd = payload["last_upd"]
        exp = _expected_labels(mkey)
        if exp is not None:
            if not exp.issubset(set(vals.keys())):
                continue  # incomplete -> skip hele set
        for label, odd in vals.items():
            out_rows.append((fx, bmid, mkey, label, odd, last_upd))
    buf.clear()


def _skip_market_by_name(bet):
    """Markten die we sowieso niet willen verwerken."""
    name = _norm(bet.get("name"))
    if not name:
        return False
    return any(k in name for k in (
        "winning margin","correct score","method of goal","goals range",
        "exact goals","exactly","goals odd/even","shots","shots on target",
        "cards","corners race","race to","clean sheet","to qualify",
        "penalties","extra time"
    ))

def _extract_corners_ou_values(fixture_id, bookmaker_id, bet, last_upd, out_rows):
    line = _find_ou_line_in_name_or_values(bet)
    if not line:
        # log één keer per uniek labelset
        _log_unknown("corners_ou", bet.get("name", ""))
        return
    for v in (bet.get("values") or []):
        lab = _norm_ou_label_with_line(v.get("value"), line)
        odd = _to_float(v.get("odd"))
        if lab and odd is not None and bookmaker_id is not None:
            out_rows.append((fixture_id, bookmaker_id, "corners_ou", lab, odd, last_upd))
        elif lab is None:
            _log_unknown("corners_ou", v.get("value", ""))

def _extract_fh_1x2_values(fixture_id, bookmaker_id, bet, last_upd, buf):
    for val in bet.get("values") or []:
        if not isinstance(val, dict):
            continue
        label = _map_1x2_value(val.get("value"))
        if not label:
            _log_unknown("fh_1x2", val.get("label", val.get("value", "")))
            continue
        odd = _to_float(val.get("odd"))
        if odd is None or bookmaker_id is None:
            continue
        _buf_add(buf, fixture_id, bookmaker_id, "fh_1x2", label, odd, last_upd)

def _is_corners_ou_market(bet):
    name = _norm(bet.get("name"))
    if not name:
        return False
    if "corner" not in name:
        return False
    # skip 1st/2nd half varianten hier; die kun je later als eigen markt toevoegen
    if any(k in name for k in ("1st half","first half","1. half","2nd half","second half","2. half")):
        return False
    if _is_result_totals_combo(bet):
        return False
    if not (_OU_PAT.search(name) or "over" in name or "under" in name or "total" in name or "totals" in name):
        return False
    line = _find_ou_line_in_name_or_values(bet)
    return line is not None


def _is_fh_1x2_market(bet):
    name = _norm(bet.get("name"))
    if not name:
        return False
    if not ("1st half" in name or "first half" in name or "1. half" in name):
        return False
    # geen corners/cards etc.
    if any(k in name for k in ("corner","card","booking","method","shot","header","penalty","freekick")):
        return False
    vals = bet.get("values") or []
    mapped = {_map_1x2_value(v.get("value")) for v in vals if isinstance(v, dict)}
    mapped.discard(None)
    return len(mapped) >= 2

_OU_LINE_PAT = re.compile(r'([0-9]+(?:[.,][0-9])?)')

def _find_ou_line_in_name_or_values(bet):
    # 1) probeer in naam
    name = _norm(bet.get("name"))
    m = _OU_LINE_PAT.search(name)
    if m:
        return m.group(1).replace(',', '.')
    # 2) anders in value labels
    for v in (bet.get("values") or []):
        s = _norm(v.get("value"))
        m = _OU_LINE_PAT.search(s)
        if m:
            return m.group(1).replace(',', '.')
    return None

def _norm_ou_label_with_line(vlabel, line):
    s = _norm(vlabel)
    if s.startswith('over') or s.startswith('o'):
        return f'over_{line}'
    if s.startswith('under') or s.startswith('u'):
        return f'under_{line}'
    return None

def _utc_date_from_iso(ts: str):
    try:
        dt = datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).date()
    except Exception:
        return None

def _to_float(x):
    try:
        if x is None:
            return None
        x = str(x).strip().replace(",", ".")
        if x == "":
            return None
        return float(x)
    except:
        return None

def _norm(s):
    try:
        return str(s).strip().lower()
    except Exception:
        return ""

def _map_1x2_value(x):
    s = _norm(x)
    if s in ("home","1","1 (home)"): return "Home"
    if s in ("draw","x","tie","equal","no winner","x (draw)"): return "Draw"
    if s in ("away","2","2 (away)"): return "Away"
    # tolerante fallback
    if "home" in s: return "Home"
    if "away" in s: return "Away"
    if "draw" in s: return "Draw"
    return None

def _is_1x2_market(bet):
    """
    Alleen de echte Match Winner markt toelaten.
    """
    bet_id = bet.get("id")
    name = _norm(bet.get("name"))
    return bet_id == BET_ID_1X2 and name == "match winner"

_FH_TEAM_TOTAL_BLOCK = re.compile(
    r"\b(home|away)\b.*\bteam\b|\bteam\b.*\b(home|away)\b|"
    r"\bteam\s*total\b|\bhome\s*team\b|\baway\s*team\b|\bteam\s*goals\b",
    re.I
)

def _is_fh_ou_market(bet):
    name = _norm(bet.get("name"))
    if not name:
        return False

    # Must be first half
    if not (("1st half" in name) or ("first half" in name) or ("1. half" in name)):
        return False

    # Must look like totals O/U
    if not (("over" in name) or ("under" in name) or ("o/u" in name) or bool(_OU_PAT.search(name))):
        return False

    # Hard block: team totals / home/away team variants
    if _FH_TEAM_TOTAL_BLOCK.search(name):
        return False

    # Must have a line in name or values
    return _find_ou_line_in_name_or_values(bet) is not None

UNKNOWN_LABELS = {}  # {market_key: set(labels)}

# suppression van bekende ruislabels per markt
_SUPPRESS_PATTERNS = {
    "1x2": [
        r"^\d+$",               # 0,3,4,5,6...
        r"^more\s+\d+$",        # "more 4"
        r"^(1st|2nd)\s+half$",  # "1st Half", "2nd Half"
        r"\bby\s+\d+\+?$",      # "1 by 4+"
        r"^(shot|header|penalty|freekick|owngoal)$",
        r"^(exactly|over|under)\s*\d+$",
    ],
    "ou_2_5": [
        r"^(home|draw|away)\s*/\s*(over|under)\s*2[.,]5$"  # result&totals combo
    ],
    "fh_1x2": [
        r"^\d+$",
        r"^more\s+\d+$",
        r"^(shot|header|penalty|freekick|owngoal)$",
    ],
    "corners_ou": [
        r"^(home|draw|away)\s*/",   # result/totals-combo bij corners
        r"^exact(ly)?\s*\d+\s*$",   # 'Exactly 5', 'exact 12', etc.
    ],
}

def _should_suppress_unknown(market_key, raw_label) -> bool:
    s = _norm_label(raw_label)
    for pat in _SUPPRESS_PATTERNS.get(market_key, []):
        if re.search(pat, s):
            return True
    return False

def _log_unknown(market_key, raw_label):
    if _should_suppress_unknown(market_key, raw_label):
        return
    label = _norm_label(raw_label) or "<empty>"
    seen = UNKNOWN_LABELS.setdefault(market_key, set())
    if label in seen:
        return
    seen.add(label)
    print(f"[odds][unknown] market={market_key} label_raw={raw_label!r}")

def _extract_1x2_values(fixture_id, bookmaker_id, bet, last_upd, buf):
    # extra safety: alleen echte Match Winner
    if bet.get("id") != BET_ID_1X2 or _norm(bet.get("name")) != "match winner":
        return

    for val in bet.get("values") or []:
        if not isinstance(val, dict):
            continue
        raw = val.get("value")
        label = _map_1x2_value(raw)
        if not label:
            if not _should_suppress_unknown("1x2", raw):
                _log_unknown("1x2", val.get("label", raw))
            continue

        odd = _to_float(val.get("odd"))
        if odd is None or bookmaker_id is None:
            continue

        _buf_add(buf, fixture_id, bookmaker_id, "1x2", label, odd, last_upd)

def _extract_fh_ou_values(fixture_id, bookmaker_id, bet, last_upd, buf):
    """
    Leest 'Goals Over/Under First Half' en slaat ALLEEN 0.5 en 1.5 op (voor nu).
    Opslag:
      market_key: fh_ou_0_5 of fh_ou_1_5
      label     : over_0.5 / under_0.5 / over_1.5 / under_1.5
    """
    for v in (bet.get("values") or []):
        if not isinstance(v, dict):
            continue

        val = v.get("value")
        odd = _to_float(v.get("odd"))
        if odd is None or bookmaker_id is None:
            continue

        s = _norm(val)
        m = _OU_LINE_PAT.search(s)
        if not m:
            continue

        line = m.group(1).replace(",", ".")
        if line not in ("0.5", "1.5"):
            continue  # breid later uit als je wil

        lab = _norm_ou_label_with_line(val, line)  # geeft over_0.5 / under_1.5 etc
        if not lab:
            continue

        market_key = f"fh_ou_{line.replace('.', '_')}"  # fh_ou_0_5 of fh_ou_1_5
        _buf_add(buf, fixture_id, bookmaker_id, market_key, lab, odd, last_upd)


# ---------- Detectie ----------
_BTTS_PAT = re.compile(r'\b(btts|both\s*teams\s*to\s*score|both\s*to\s*score)\b', re.I)
_OU_PAT   = re.compile(r'\b(over/under|totals?)\b', re.I)
_DNB_PAT  = re.compile(r'\b(draw\s*no\s*bet|dnb)\b', re.I)
_AH_PAT   = re.compile(r'\b(asian\s*handicap|ah)\b', re.I)

def _is_btts_market(bet):
    return bool(_BTTS_PAT.search(_norm(bet.get("name"))))

def _is_result_totals_combo(bet):
    name = _norm(bet.get("name"))
    if any(k in name for k in ("result & total", "result and total", "1x2 & over/under", "combo")):
        return True
    for v in (bet.get("values") or []):
        s = _norm(v.get("value"))
        if "/" in s and ("over" in s or "under" in s) and re.search(r'2[.,]5', s):
            return True
    return False

def _is_ou_2_5_market(bet):
    """
    Alleen de echte Goals Over/Under markt toelaten.
    Daarna filteren we in de extractor alleen de 2.5-regels eruit.
    """
    bet_id = bet.get("id")
    name = _norm(bet.get("name"))
    return bet_id == BET_ID_OU and name == "goals over/under"

def _is_dnb_market(bet):
    return bool(_DNB_PAT.search(_norm(bet.get("name"))))

def _is_ah_market(bet):
    bet_id = bet.get("id")
    name = _norm(bet.get("name"))
    return bet_id == BET_ID_AH and name == "asian handicap"

# ---------- Normalisatie labels ----------
def _norm_btts_label(vlabel: str) -> str | None:
    s = _norm(vlabel)
    if re.search(r'\b(yes|ja)\b', s): return 'yes'
    if re.search(r'\b(no|nee)\b', s): return 'no'
    return None

def _norm_ou_2_5_label(vlabel: str) -> str | None:
    s = _norm(vlabel)
    if s.startswith('over') or s in ('o','o 2.5','o2.5','over 2.5'):  return 'over_2.5'
    if s.startswith('under') or s in ('u','u 2.5','u2.5','under 2.5'): return 'under_2.5'
    return None

def _norm_dnb_label(vlabel: str) -> str | None:
    s = _norm(vlabel)
    if s in ('home','1','1 (home)') or 'home' in s: return 'home'
    if s in ('away','2','2 (away)') or 'away' in s: return 'away'
    return None

_AH_LINE_PAT = re.compile(r'([+-]?\d+(?:[.,]\d+)?)')

def _norm_ah_label(vlabel: str) -> str | None:
    s = _norm(vlabel)
    side = 'home' if re.search(r'\b(home|^1$)\b', s) else ('away' if re.search(r'\b(away|^2$)\b', s) else None)
    m = _AH_LINE_PAT.search(s)
    if not side or not m:
        return None
    line = m.group(1).replace(',', '.')
    if not line.startswith(('+','-')):
        line = f'+{line}'
    return f'{side} {line}'

# ---------- Extractors ----------
def _extract_btts_values(fixture_id, bookmaker_id, bet, last_upd, buf):
    for v in (bet.get("values") or []):
        lab = _norm_btts_label(v.get("value"))
        odd = _to_float(v.get("odd"))
        if lab and odd is not None and bookmaker_id is not None:
            _buf_add(buf, fixture_id, bookmaker_id, "btts", lab, odd, last_upd)
        elif lab is None:
            _log_unknown("btts", v.get("value", ""))

def _extract_ou_2_5_values(fixture_id, bookmaker_id, bet, last_upd, buf):
    """
    Alleen Over 2.5 en Under 2.5 bewaren uit de echte Goals Over/Under markt.
    """
    if bet.get("id") != BET_ID_OU or _norm(bet.get("name")) != "goals over/under":
        return

    for v in (bet.get("values") or []):
        raw = v.get("value")
        odd = _to_float(v.get("odd"))

        if odd is None or bookmaker_id is None:
            continue

        s = _norm(raw)

        if s == "over 2.5":
            _buf_add(buf, fixture_id, bookmaker_id, "ou_2_5", "over_2.5", odd, last_upd)
        elif s == "under 2.5":
            _buf_add(buf, fixture_id, bookmaker_id, "ou_2_5", "under_2.5", odd, last_upd)

def _extract_dnb_values(fixture_id, bookmaker_id, bet, last_upd, buf):
    for v in (bet.get("values") or []):
        lab = _norm_dnb_label(v.get("value"))
        odd = _to_float(v.get("odd"))
        if lab and odd is not None and bookmaker_id is not None:
            _buf_add(buf, fixture_id, bookmaker_id, "dnb", lab, odd, last_upd)
        elif lab is None:
            _log_unknown("dnb", v.get("value", ""))

def _extract_ah_values(fixture_id, bookmaker_id, bet, last_upd, out_rows):
    """
    Alleen de echte Asian Handicap markt verwerken.
    """
    if bet.get("id") != BET_ID_AH or _norm(bet.get("name")) != "asian handicap":
        return

    for v in (bet.get("values") or []):
        lab = _norm_ah_label(v.get("value"))
        odd = _to_float(v.get("odd"))
        if lab and odd is not None and bookmaker_id is not None:
            out_rows.append((fixture_id, bookmaker_id, "ah", lab, odd, last_upd))
        elif lab is None:
            _log_unknown("ah", v.get("value", ""))

# ========= Stap 4: fixtures =========

def load_fixtures_for_policy(past_days=HORIZON_PAST_DAYS, future_days=HORIZON_FUTURE_DAYS, season=SEASON):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT league_id FROM league_policy WHERE policy='include'")
        allowed_ids = {r[0] for r in cur.fetchall()}

    fixtures_rows, venues_rows2 = [], []

    mode = FIXTURES_MODE
    if mode == "DATE_GLOBAL" and len(allowed_ids) < 50:
        mode = "BY_LEAGUE"

    if mode == "BY_LEAGUE":
        date_from = (datetime.now(UTC).date() - timedelta(days=past_days)).isoformat()
        date_to = (datetime.now(UTC).date() + timedelta(days=future_days)).isoformat()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT league_id, country, name FROM leagues
                WHERE league_id = ANY(%s)
                ORDER BY country, name
            """, (list(allowed_ids),))
            leagues = cur.fetchall()
        total = len(leagues)
        for idx, (lg, country, name) in enumerate(leagues, start=1):
            league_season = get_season_for_league(lg, fallback_season=season)
            tag = f"{idx}/{total} league={lg} {country} - {name} season={league_season}"
            try:
                data = get_json(
                    "/fixtures",
                    params={"league": lg, "season": league_season, "from": date_from, "to": date_to, "timezone": "UTC"},
                    cache_ttl=1800
                )
                resp = data.get("response", [])
                kept = 0
                for item in resp:
                    vrow = _collect_fixture_venue(item)
                    if vrow: venues_rows2.append(vrow)
                    fixtures_rows.append(_row_from_fixture_item(item))
                    kept += 1
                print(f"[fixtures] {tag} fixtures_total={len(resp)} kept={kept}")
            except Exception as e:
                print(f"[fixtures][WARN] {tag} error: {e}")
            if idx % 2 == 0 or idx == total:
                if venues_rows2:
                    bulk_upsert("venues", ["venue_id","name","address","city","capacity","surface","image"], venues_rows2, ["venue_id"])
                    venues_rows2.clear()
                _upsert_fixtures(fixtures_rows)
                fixtures_rows.clear()
    else:
        dates = _date_list(past_days, future_days)
        total = len(dates)
        for idx, dt in enumerate(dates, start=1):
            try:
                data = get_json("/fixtures", params={"date": dt, "season": season, "timezone": "UTC"}, cache_ttl=1800)
                resp = data.get("response", [])
                total_api, kept = len(resp), 0
                for item in resp:
                    if (item.get("league") or {}).get("id") in allowed_ids:
                        vrow = _collect_fixture_venue(item)
                        if vrow: venues_rows2.append(vrow)
                        fixtures_rows.append(_row_from_fixture_item(item))
                        kept += 1
                if idx == 1 or idx % 3 == 0:
                    print(f"[fixtures] {idx}/{total} {dt} fixtures_total={total_api} kept={kept}")
            except Exception as e:
                print(f"[fixtures][WARN] date={dt} error: {e}")
            if idx % 2 == 0 or idx == total:
                if venues_rows2:
                    bulk_upsert("venues", ["venue_id","name","address","city","capacity","surface","image"], venues_rows2, ["venue_id"])
                    venues_rows2.clear()
                _upsert_fixtures(fixtures_rows)
                fixtures_rows.clear()
    print("[fixtures] klaar.")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT fixture_id
            FROM fixtures
            WHERE league_id = ANY(%s)
              AND date_utc >= NOW() - INTERVAL %s
              AND date_utc <= NOW() + INTERVAL %s
        """, (list(allowed_ids), f"{past_days} days", f"{future_days} days"))
        return [r[0] for r in cur.fetchall()]

# ========= Stap 5: odds =========

def _persist_odds_bulk(rows_to_upsert):
    """
    rows_to_upsert: list of tuples (fixture_id, bookmaker_id, market_key, label, odd, last_update)
    - Dedupe binnen de batch op (fixture_id, bookmaker_id, market_key, label) zodat
      één INSERT ... ON CONFLICT geen rij twee keer raakt.
    - Snapshots: ook dedupe, anders kan de (fixture_id, bookmaker_id, market_key, label, captured_at)
      PK botsen omdat NOW() per statement gelijk is.
    """
    if not rows_to_upsert:
        return

    # ---- 1) Dedup: last-write-wins binnen deze batch ----
    # key: (fixture_id, bookmaker_id, market_key, label)
    uniq = {}
    for r in rows_to_upsert:
        # r = (fx, bmid, mkey, label, odd, last_update)
        key = (r[0], r[1], r[2], r[3])
        uniq[key] = r  # laatste wint

    deduped = list(uniq.values())
    if not deduped:
        return

    # Voor snapshots gebruiken we dezelfde deduped set
    snapshot_rows = [(r[0], r[1], r[2], r[3], r[4]) for r in deduped]

    with get_conn() as conn, conn.cursor() as cur:
        # Upsert naar odds_values
        sql_up = """
        INSERT INTO odds_values (fixture_id, bookmaker_id, market_key, label, odd, last_update)
        VALUES %s
        ON CONFLICT (fixture_id, bookmaker_id, market_key, label)
        DO UPDATE SET odd=EXCLUDED.odd, last_update=EXCLUDED.last_update
        """
        for i in range(0, len(deduped), 5000):
            pgx.execute_values(cur, sql_up, deduped[i:i+5000],
                               template="(%s,%s,%s,%s,%s,%s)", page_size=1000)

        # Append snapshots (captured_at = NOW())
        # NB: omdat NOW() binnen 1 statement gelijk is, moet dit óók deduped zijn.
        sql_sn = """
        INSERT INTO odds_values_snapshots (fixture_id, bookmaker_id, market_key, label, odd)
        VALUES %s
        """
        for i in range(0, len(snapshot_rows), 5000):
            pgx.execute_values(cur, sql_sn, snapshot_rows[i:i+5000],
                               template="(%s,%s,%s,%s,%s)", page_size=1000)

        conn.commit()


def load_odds_for_fixtures(fixture_ids, max_cache_ttl=300):
    if not fixture_ids:
        print("[odds] no fixtures")
        return

    fetched = 0
    bm_rows = {}
    odds_rows = []
    fixtures_with_any_odds = 0
    buf = {}

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT fixture_id, league_id, season, (date_utc AT TIME ZONE 'UTC')::date AS d
            FROM fixtures
            WHERE fixture_id = ANY(%s)
        """, (list(fixture_ids),))
        rows = cur.fetchall()

    groups = defaultdict(list)
    today = datetime.now(timezone.utc).date()
    for fx, lg, s, d in rows:
        if d and (d >= today - timedelta(days=7)) and (d <= today + timedelta(days=14)):
            groups[(lg, s, d)].append(fx)
        else:
            fetched += 1

    total_groups = len(groups)
    for gi, ((lg, s, d), fxs) in enumerate(groups.items(), start=1):
        try:
            data = get_json("/odds", params={
                "league": lg, "season": s, "date": d.isoformat(), "timezone": "UTC"
            }, cache_ttl=max_cache_ttl)
        except Exception as e:
            print(f"[odds][HTTP] lg={lg} s={s} d={d} error: {e}")
            data = {}

        resp = (data or {}).get("response") or []
        before = len(odds_rows)

        # -------------------------------
        # VERWERK RESP (MOET BINNEN LOOP)
        # -------------------------------
        for it in resp:
            fx = (it.get("fixture") or {}).get("id")
            if fx not in fxs:
                continue

            bookmakers = it.get("bookmakers") or []
            for bm in bookmakers:
                bmid, bmname = bm.get("id"), bm.get("name")

                # Alleen Bet365 en Pinnacle
                if bmid not in ALLOWED_BOOKMAKERS:
                    continue

                if bmid and bmname:
                    bm_rows[bmid] = bmname

                # BUF per bookmaker+fixture
                buf.clear()

                for bet in bm.get("bets") or []:
                    if _skip_market_by_name(bet):
                        continue

                    last_upd = bet.get("last_update") or it.get("update") or None

                    if _is_1x2_market(bet):
                        _extract_1x2_values(fx, bmid, bet, last_upd, buf)
                        continue

                    if _is_ah_market(bet):
                        _extract_ah_values(fx, bmid, bet, last_upd, odds_rows)
                        continue

                    if _is_ou_2_5_market(bet):
                        _extract_ou_2_5_values(fx, bmid, bet, last_upd, buf)
                        continue

                # flush gated sets naar odds_rows
                _flush_buf_to_rows(buf, odds_rows)

        # ---- stats/flush per group ----
        added = len(odds_rows) - before
        if added > 0:
            fixtures_with_any_odds += 1
        fetched += len(fxs)

        # Periodiek flushen
        if gi % 5 == 0:
            if bm_rows:
                bm_upsert = [(k, v) for k, v in bm_rows.items()]
                bulk_upsert("odds_bookmakers", ["bookmaker_id","name"], bm_upsert, ["bookmaker_id"])
                bm_rows.clear()
            if odds_rows:
                rows_flush = [r for r in odds_rows if r[1] is not None and r[4] is not None]
                if rows_flush:
                    _persist_odds_bulk(rows_flush)
                    print(f"[odds] flush groups upserted_rows={len(rows_flush)}(+snapshots)")
                odds_rows.clear()

        if gi == 1 or gi % 5 == 0:
            print(f"[odds] group {gi}/{total_groups} lg={lg} s={s} d={d} added+={added}")


    # Final flush
    if bm_rows:
        bm_upsert = [(k, v) for k, v in bm_rows.items()]
        bulk_upsert("odds_bookmakers", ["bookmaker_id","name"], bm_upsert, ["bookmaker_id"])
    if odds_rows:
        rows_flush = [r for r in odds_rows if r[1] is not None and r[4] is not None]
        if rows_flush:
            _persist_odds_bulk(rows_flush)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM odds_bookmakers"); bm_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT fixture_id) FROM odds_values"); fx_with_odds = cur.fetchone()[0]
        cur.execute("SELECT market_key, COUNT(*) FROM odds_values GROUP BY market_key ORDER BY 2 DESC"); per_market = cur.fetchall()

    print(f"[odds] fixtures_touched~={fetched} fixtures_with_any_odds={fixtures_with_any_odds} bookmakers={bm_count}")
    print(f"[odds] fixtures_in_db_with_any_odds={fx_with_odds} per_market={per_market}")


# ========= NEW: Events & Team Stats =========

def _finished_between_ids(d_from: datetime, d_to: datetime):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT fixture_id
            FROM fixtures
            WHERE (date_utc::date BETWEEN %s AND %s)
              AND COALESCE(status_short,'') IN ('FT','AET','PEN','AWD','WO')
        """, (d_from.date(), d_to.date()))
        return [r[0] for r in cur.fetchall()]

def load_events_for_fixtures(fixture_ids, cache_ttl=24*3600):
    """
    Laadt /fixtures/events en schrijft in batches met dedupe op de conflict-sleutel:
    (fixture_id, event_time, event_extra0, COALESCE(team_id,0), COALESCE(player_id,0), type, detail)
    """
    if not fixture_ids:
        print("[events] no finished fixtures in window")
        return

    rows = []
    batch_fx = 0
    BATCH_EVERY_FX = 250       # flush na X fixtures
    BATCH_MIN_ROWS = 2000      # of zodra er >= X event-rijen zijn

    def flush():
        nonlocal rows
        if not rows:
            return

        # --- DEDUPE op conflict-sleutel binnen deze flush ---
        # rows: (fixture_id, event_time, event_extra, event_extra0,
        #        team_id, player_id, player_name, assist_id, assist_name,
        #        type, detail, comments)
        uniq = {}
        for r in rows:
            fixture_id, event_time, event_extra, event_extra0, team_id, player_id, player_name, assist_id, assist_name, etype, detail, comments = r

            # PK-kolommen mogen niet NULL zijn -> coalesce in de sleutel
            k_event_time = -1 if event_time is None else event_time
            k_team_id0   = team_id if team_id is not None else 0
            k_player_id0 = player_id if player_id is not None else 0
            k_type       = etype or ""
            k_detail     = detail or ""

            key = (fixture_id, k_event_time, event_extra0, k_team_id0, k_player_id0, k_type, k_detail)

            # Bewaar een "genormaliseerde" rij die overeenkomt met de sleutel (geen NULLs in PK-velden)
            norm_row = (
                fixture_id,
                k_event_time,
                event_extra,
                event_extra0,
                team_id,
                player_id,
                player_name,
                assist_id,
                assist_name,
                k_type,
                k_detail,
                comments
            )
            uniq[key] = norm_row  # laatste wint

        deduped = list(uniq.values())

        # --- INSERT (zonder generated kolommen) + ON CONFLICT op de PK met team_id0/player_id0 ---
        cols = [
            "fixture_id","event_time","event_extra","event_extra0",
            "team_id","player_id","player_name",
            "assist_id","assist_name","type","detail","comments"
        ]
        placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
        sql = f"""
        INSERT INTO fixture_events ({", ".join(cols)})
        VALUES %s
        ON CONFLICT (fixture_id, event_time, event_extra0, team_id0, player_id0, type, detail)
        DO UPDATE SET
            event_extra = EXCLUDED.event_extra,
            team_id     = EXCLUDED.team_id,
            player_id   = EXCLUDED.player_id,
            player_name = EXCLUDED.player_name,
            assist_id   = EXCLUDED.assist_id,
            assist_name = EXCLUDED.assist_name,
            comments    = EXCLUDED.comments
        """

        # Chunked write om veilig te blijven
        CHUNK = 2000
        with get_conn() as conn, conn.cursor() as cur:
            for i in range(0, len(deduped), CHUNK):
                batch = deduped[i:i+CHUNK]
                pgx.execute_values(cur, sql, batch, template=placeholders, page_size=500)
            conn.commit()

        print(f"[events] upserted_rows+={len(deduped)} (flush)")
        rows = []

    # ---- API calls + verzamelen ----
    for i, fx in enumerate(fixture_ids, start=1):
        try:
            data = get_json("/fixtures/events", params={"fixture": fx}, cache_ttl=cache_ttl)
            for ev in (data or {}).get("response", []):
                t    = ev.get("time")    or {}
                tm   = ev.get("team")    or {}
                pl   = ev.get("player")  or {}
                asst = ev.get("assist")  or {}

                e_time  = t.get("elapsed")
                if e_time is None:
                    e_time = -1  # PK-kolom -> geen NULL

                e_extra  = t.get("extra")
                e_extra0 = int(e_extra) if (e_extra is not None) else 0

                etype  = (ev.get("type")   or "")
                detail = (ev.get("detail") or "")

                rows.append((
                    fx,
                    e_time, e_extra, e_extra0,
                    tm.get("id"),
                    pl.get("id"), pl.get("name"),
                    asst.get("id"), asst.get("name"),
                    etype, detail, ev.get("comments")
                ))
        except Exception as e:
            print(f"[events][WARN] fixture={fx} error: {e}")

        batch_fx += 1
        if (batch_fx >= BATCH_EVERY_FX) or (len(rows) >= BATCH_MIN_ROWS):
            flush()
            print(f"[events] progress {i}/{len(fixture_ids)} fixtures processed")
            batch_fx = 0

    flush()  # final


def load_team_stats_for_fixtures(fixture_ids, cache_ttl=24*3600):
    if not fixture_ids:
        print("[stats] no finished fixtures in window")
        return

    rows = []
    batch_fx = 0
    BATCH_EVERY_FX = 250
    BATCH_MIN_ROWS = 1000  # teamstats = 2 rijen per fixture

    def flush():
        nonlocal rows
        if not rows:
            return
        bulk_upsert("fixture_statistics_team", ["fixture_id","team_id","stats"], rows, ["fixture_id","team_id"])
        print(f"[stats] upserted_rows+={len(rows)} (flush)")
        rows = []

    for i, fx in enumerate(fixture_ids, start=1):
        try:
            data = get_json("/fixtures/statistics", params={"fixture": fx}, cache_ttl=cache_ttl)
            for it in (data or {}).get("response", []):
                team = (it.get("team") or {}).get("id")
                stats = it.get("statistics") or []
                if team:
                    rows.append((fx, team, json.dumps(stats)))
        except Exception as e:
            print(f"[stats][WARN] fixture={fx} error: {e}")

        batch_fx += 1
        if (batch_fx >= BATCH_EVERY_FX) or (len(rows) >= BATCH_MIN_ROWS):
            flush()
            print(f"[stats] progress {i}/{len(fixture_ids)} fixtures processed")
            batch_fx = 0

    flush()  # final


# ========= HIST BACKFILL =========

def _date_windows(start_date, end_date, window_days=30):
    cur = start_date
    delta = timedelta(days=window_days-1)
    while cur <= end_date:
        w_start = cur
        w_end = min(end_date, cur + delta)
        yield (w_start, w_end)
        cur = w_end + timedelta(days=1)

def _allowed_league_ids(filtered_league_id=None, name_contains=None):
    with get_conn() as conn, conn.cursor() as cur:
        if filtered_league_id:
            try:
                lid = int(filtered_league_id)
            except:
                return []
            cur.execute("SELECT 1 FROM leagues WHERE league_id=%s", (lid,))
            return [lid] if cur.fetchone() else []
        elif name_contains:
            cur.execute("""
                SELECT l.league_id
                FROM league_policy lp JOIN leagues l ON l.league_id=lp.league_id
                WHERE lp.policy='include' AND LOWER(l.name) LIKE %s
            """, (f"%{name_contains.lower()}%",))
        else:
            cur.execute("SELECT league_id FROM league_policy WHERE policy='include'")
        return [r[0] for r in cur.fetchall()]

def _fixtures_finished_without_odds(league_id, d_from, d_to):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            WITH window_fixtures AS (
                SELECT f.fixture_id, f.date_utc::date AS d
                FROM fixtures f
                WHERE f.league_id=%s
                  AND f.date_utc::date BETWEEN %s AND %s
                  AND COALESCE(f.status_short,'') IN ('FT','AET','PEN','AWD','WO')
            )
            SELECT wf.fixture_id
            FROM window_fixtures wf
            LEFT JOIN (
                SELECT DISTINCT fixture_id FROM odds_values
                WHERE market_key IN ('1x2','ah','ou_2_5')
            ) ov ON ov.fixture_id = wf.fixture_id
            WHERE ov.fixture_id IS NULL
        """, (league_id, d_from, d_to))
        return [r[0] for r in cur.fetchall()]

def _fetch_window_with_halving(league_id, start_d, end_d, season):
    stack = [(start_d, end_d)]
    out = []
    while stack:
        a, b = stack.pop()
        items = get_all_pages(
            "/fixtures",
            params={"league": league_id, "season": season, "from": a.isoformat(), "to": b.isoformat(), "timezone": "UTC"},
            cache_ttl=24*3600
        )
        if items:
            out.extend(items)
        else:
            if (b - a).days > 1:
                mid = a + (b - a) // 2
                stack.append((a, mid))
                stack.append((mid + timedelta(days=1), b))
    return out

def hist_backfill_run():
    start_s = os.getenv("BACKFILL_START")
    end_s = os.getenv("BACKFILL_END")
    if not start_s or not end_s:
        raise RuntimeError("HIST_BACKFILL vereist BACKFILL_START en BACKFILL_END (YYYY-MM-DD).")

    window_days = int(os.getenv("BACKFILL_WINDOW_DAYS", "30"))
    league_id_filter = os.getenv("BACKFILL_LEAGUE_ID")
    league_name_sub = os.getenv("BACKFILL_LEAGUE_NAME")
    season_for_backfill = int(os.getenv("BACKFILL_SEASON", str(SEASON)))
    fetch_odds_flag = os.getenv("BACKFILL_FETCH_ODDS", "1")
    fetch_odds = (fetch_odds_flag != "0")

    start_d = datetime.strptime(start_s, "%Y-%m-%d").date()
    end_d = datetime.strptime(end_s, "%Y-%m-%d").date()
    if end_d < start_d:
        raise RuntimeError("BACKFILL_END mag niet vóór BACKFILL_START liggen.")

    league_ids = _allowed_league_ids(league_id_filter, league_name_sub)
    if not league_ids:
        print("[hist] geen leagues gevonden voor deze scope.")
        return

    # plan print
    print("== HIST_BACKFILL plan ==")
    print(f"Leagues : {len(league_ids)} (filter id={league_id_filter} name~{league_name_sub})")
    print(f"Windows : {window_days} dagen per venster")
    print(f"Datumrange : {start_d} t/m {end_d}")
    print(f"Season : {season_for_backfill}")
    print(f"Fetch odds : {'JA' if fetch_odds else 'NEE'}")

    def _league_name(lid):
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT name, country FROM leagues WHERE league_id=%s", (lid,))
                r = cur.fetchone()
                if r:
                    return f"{r[1]} – {r[0]}"
        except:
            pass
        return f"league_id={lid}"

    # MAX_WORKERS via ENV (default 1 = sequentieel)
    max_workers = int(os.getenv("MAX_WORKERS", "1"))
    print(f"[hist] using MAX_WORKERS={max_workers}")

    if max_workers <= 1:
        # sequentieel
        for lg in league_ids:
            _process_league_backfill(
                lg, start_d, end_d, window_days, season_for_backfill,
                fetch_odds, LOAD_EVENTS, LOAD_STATS, _league_name
            )
    else:
        # parallel per league (eigen DB per thread, shared rate-limiter met lock)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for lg in league_ids:
                futs.append(ex.submit(
                    _process_league_backfill,
                    lg, start_d, end_d, window_days, season_for_backfill,
                    fetch_odds, LOAD_EVENTS, LOAD_STATS, _league_name
                ))
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    print(f"[hist][worker][ERROR] {e}")

    with get_conn() as conn:
        for lg in league_ids:
            for w_start, w_end in _date_windows(start_d, end_d, window_days):
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM backfill_progress
                        WHERE league_id=%s AND window_start=%s AND window_end=%s
                    """, (lg, w_start, w_end))
                    if cur.fetchone():
                        print(f"[hist] league={lg} {w_start}..{w_end} al gedaan -> skip")
                        continue

                league_season = get_season_for_league(lg, fallback_season=season_for_backfill)
                resp = []

                try:
                    probe_params = {
                        "league": lg,
                        "season": league_season,
                        "from": w_start.isoformat(),
                        "to": w_end.isoformat(),
                        "timezone": "UTC",
                    }
                    probe = get_json("/fixtures", params=probe_params, cache_ttl=0)
                    if (probe.get("response") or []):
                        resp = probe.get("response") or []
                    if not resp:
                        resp = _fetch_window_with_halving(lg, w_start, w_end, league_season)

                    print(f"[hist] {_league_name(lg)} {w_start}..{w_end} season={league_season} resp={len(resp)}")
                except Exception as e:
                    print(f"[hist][direct][WARN] league={lg} {w_start}..{w_end} error: {e}")
                    resp = []

                if not resp:
                    gathered, dt = [], w_start
                    fetched_sum = 0
                    kept_sum = 0
                    while dt <= w_end:
                        try:
                            day_items = get_all_pages(
                                "/fixtures",
                                params={"date": dt.isoformat(), "season": league_season, "timezone": "UTC"},
                                cache_ttl=0
                            )
                            fetched_sum += len(day_items)
                            for it in day_items:
                                if (it.get("league") or {}).get("id") == lg:
                                    gathered.append(it); kept_sum += 1
                        except Exception as e:
                            print(f"[hist][day][WARN] league={lg} date={dt} error: {e}")
                        dt += timedelta(days=1)
                    resp = gathered
                    print(f"[hist][fallback-scan] {_league_name(lg)} {w_start}..{w_end} fetched_total={fetched_sum} kept_for_league={kept_sum}")

                venues_rows, fixtures_rows = [], []
                for item in resp:
                    vrow = _collect_fixture_venue(item)
                    if vrow: venues_rows.append(vrow)
                    fixtures_rows.append(_row_from_fixture_item(item))

                if venues_rows:
                    bulk_upsert("venues", ["venue_id","name","address","city","capacity","surface","image"], venues_rows, ["venue_id"])
                _upsert_fixtures(fixtures_rows)

                tried = 0
                if fetch_odds and fixtures_rows:
                    # 1) Kandidaten: afgewerkte fixtures zonder odds in dit venster
                    fx_targets = _fixtures_finished_without_odds(lg, w_start, w_end)

                    # 2) Beperk tot odds-venster (-7d..+14d rond vandaag)
                    with get_conn() as _c:
                        with _c.cursor() as _cur:
                            _cur.execute("""
                                SELECT fixture_id, (date_utc AT TIME ZONE 'UTC')::date AS d
                                FROM fixtures
                                WHERE fixture_id = ANY(%s)
                            """, (fx_targets,))
                            today = datetime.now(timezone.utc).date()
                            fx_targets = [
                                fid for fid, d in _cur.fetchall()
                                if d and (d >= today - timedelta(days=7)) and (d <= today + timedelta(days=14))
                            ]

                    tried = len(fx_targets)

                    # 3) Odds laden voor de overgebleven fixtures
                    if fx_targets:
                        load_odds_for_fixtures(fx_targets, max_cache_ttl=600)
                else:
                    tried = 0


                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(*) FROM odds_values ov
                        JOIN fixtures f ON f.fixture_id=ov.fixture_id
                        WHERE f.league_id=%s AND f.date_utc::date BETWEEN %s AND %s
                    """, (lg, w_start, w_end))
                    total_odds_rows = cur.fetchone()[0] or 0
                    cur.execute("""
                        INSERT INTO backfill_progress(league_id, window_start, window_end, fixtures_tried, odds_calls, odds_rows)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (league_id, window_start, window_end) DO NOTHING
                    """, (lg, w_start, w_end, len(fixtures_rows), tried, total_odds_rows))
                    conn.commit()

                # NEW: meteen events/stats voor dit venster (afgelopen fixtures)
                if LOAD_EVENTS or LOAD_STATS:
                    d_from = w_start
                    d_to   = w_end
                    finished_ids = _finished_between_ids(datetime.combine(d_from, datetime.min.time()), datetime.combine(d_to, datetime.min.time()))
                    if LOAD_EVENTS:
                        load_events_for_fixtures(finished_ids, cache_ttl=7*24*3600)
                    if LOAD_STATS:
                        load_team_stats_for_fixtures(finished_ids, cache_ttl=7*24*3600)

                print(f"[hist] league={lg} {w_start}..{w_end} fixtures_upserted={len(fixtures_rows)} odds_rows_total={total_odds_rows}")

def _process_league_backfill(lg, start_d, end_d, window_days, season_for_backfill, fetch_odds, load_events, load_stats, league_label_func):
    # Elke worker gebruikt eigen DB-conn per stap
    for w_start, w_end in _date_windows(start_d, end_d, window_days):
        # skip als venster al gedaan is
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM backfill_progress
                WHERE league_id=%s AND window_start=%s AND window_end=%s
            """, (lg, w_start, w_end))
            if cur.fetchone():
                print(f"[hist] league={lg} {w_start}..{w_end} al gedaan -> skip")
                continue

        # --- fixtures halen (direct window → halvings → fallback dag-scan) ---
        league_season = get_season_for_league(lg, fallback_season=season_for_backfill)
        resp = []

        try:
            probe_params = {
                "league": lg,
                "season": league_season,
                "from": w_start.isoformat(),
                "to": w_end.isoformat(),
                "timezone": "UTC",
            }
            probe = get_json("/fixtures", params=probe_params, cache_ttl=0)
            if (probe.get("response") or []):
                resp = probe.get("response") or []
            if not resp:
                resp = _fetch_window_with_halving(lg, w_start, w_end, league_season)
                
            print(f"[hist] {league_label_func(lg)} {w_start}..{w_end} season={league_season} resp={len(resp)}")
        except Exception as e:
            print(f"[hist][direct][WARN] league={lg} {w_start}..{w_end} error: {e}")
            resp = []

        if not resp:
            gathered, dt = [], w_start
            fetched_sum = 0
            kept_sum = 0
            while dt <= w_end:
                try:
                    day_items = get_all_pages(
                        "/fixtures",
                        params={"date": dt.isoformat(), "season": league_season, "timezone": "UTC"},
                        cache_ttl=0
                    )
                    fetched_sum += len(day_items)
                    for it in day_items:
                        if (it.get("league") or {}).get("id") == lg:
                            gathered.append(it); kept_sum += 1
                except Exception as e:
                    print(f"[hist][day][WARN] league={lg} date={dt} error: {e}")
                dt += timedelta(days=1)
            resp = gathered
            print(f"[hist][fallback-scan] {league_label_func(lg)} {w_start}..{w_end} fetched_total={fetched_sum} kept_for_league={kept_sum}")

        venues_rows, fixtures_rows = [], []
        for item in resp:
            vrow = _collect_fixture_venue(item)
            if vrow: venues_rows.append(vrow)
            fixtures_rows.append(_row_from_fixture_item(item))
        if venues_rows:
            bulk_upsert("venues", ["venue_id","name","address","city","capacity","surface","image"], venues_rows, ["venue_id"])
        _upsert_fixtures(fixtures_rows)

        # odds (alleen binnen korte odds-venster)
        tried = 0
        if fetch_odds and fixtures_rows:
            fx_targets = _fixtures_finished_without_odds(lg, w_start, w_end)
            with get_conn() as _c:
                with _c.cursor() as _cur:
                    _cur.execute("""
                        SELECT fixture_id, (date_utc AT TIME ZONE 'UTC')::date AS d
                        FROM fixtures
                        WHERE fixture_id = ANY(%s)
                    """, (fx_targets,))
                    today = datetime.now(timezone.utc).date()
                    fx_targets = [fid for fid, d in _cur.fetchall()
                                  if d and (d >= today - timedelta(days=7)) and (d <= today + timedelta(days=14))]
            tried = len(fx_targets)
            if fx_targets:
                load_odds_for_fixtures(fx_targets, max_cache_ttl=600)

        # progress markeren
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM odds_values ov
                JOIN fixtures f ON f.fixture_id=ov.fixture_id
                WHERE f.league_id=%s AND f.date_utc::date BETWEEN %s AND %s
            """, (lg, w_start, w_end))
            total_odds_rows = cur.fetchone()[0] or 0
            cur.execute("""
                INSERT INTO backfill_progress(league_id, window_start, window_end, fixtures_tried, odds_calls, odds_rows)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (league_id, window_start, window_end) DO NOTHING
            """, (lg, w_start, w_end, len(fixtures_rows), tried, total_odds_rows))
            conn.commit()

        # events + stats
        if load_events or load_stats:
            finished_ids = _finished_between_ids(
                datetime.combine(w_start, datetime.min.time()),
                datetime.combine(w_end, datetime.min.time())
            )
            if load_events:
                load_events_for_fixtures(finished_ids, cache_ttl=7*24*3600)
            if load_stats:
                load_team_stats_for_fixtures(finished_ids, cache_ttl=7*24*3600)

        print(f"[hist] league={lg} {w_start}..{w_end} fixtures_upserted={len(fixtures_rows)} odds_rows_total={total_odds_rows}")



# ========= Orchestrator =========

def main():
    ensure_schema()
    cfg = get_mode_config()
    _print_plan(cfg)

    if RUN_MODE == "HIST_BACKFILL":
        if cfg["do_step_1"]:
            print("== Step 1: leagues & seasons ==")
            load_leagues_and_seasons(base_season=SEASON)
        if cfg["do_step_2"]:
            print("== Step 2: policy (whitelist via league_id) ==")
            build_league_policy_include_only(season_for_policy=SEASON)
        print("== HIST_BACKFILL ==")
        hist_backfill_run()
        if cfg["do_clean"]:
            cleanup_excluded_data()
        print("Klaar. OK")
        return

    if RUN_MODE == "CLEAN":
        cleanup_excluded_data()
        print("Klaar. OK")
        return

    if cfg["do_step_1"]:
        print("== Step 1: leagues & seasons ==")
        load_leagues_and_seasons(base_season=SEASON)

    if cfg["do_step_2"]:
        print("== Step 2: policy (whitelist via league_id) ==")
        build_league_policy_include_only(season_for_policy=SEASON)

    if cfg["do_clean"]:
        cleanup_excluded_data()

    if cfg["do_step_3"]:
        print("== Step 3: teams & venues ==")
        load_teams_for_policy(season=SEASON)

    upcoming = []
    if cfg["do_step_4"]:
        print("== Step 4: fixtures (allowed only) ==")
        upcoming = load_fixtures_for_policy(
            past_days=cfg["past_days"],
            future_days=cfg["future_days"],
            season=SEASON
        )

    if cfg["do_step_5"]:
        print("== Step 5: odds (Bet365 + Pinnacle | 1X2 + AH + OU2.5) ==")
        upcoming = list({fx for fx in upcoming if fx})
        load_odds_for_fixtures(upcoming)

    # NEW: Events & Team stats voor recent afgewerkte fixtures (normale runs)
    now = datetime.now(timezone.utc)

    if cfg["events_on"]:
        print("== Step 6: events (finished fixtures) ==")
        d_from = now - timedelta(days=cfg["events_days"])
        finished_ids = _finished_between_ids(d_from, now)
        load_events_for_fixtures(finished_ids, cache_ttl=7*24*3600)

    if cfg["stats_on"]:
        print("== Step 7: team stats (finished fixtures) ==")
        d_from = now - timedelta(days=cfg["stats_days"])
        finished_ids = _finished_between_ids(d_from, now)
        load_team_stats_for_fixtures(finished_ids, cache_ttl=7*24*3600)

    print("Klaar. OK")

if __name__ == "__main__":
    main()
