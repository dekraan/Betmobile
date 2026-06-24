# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import requests
import time
from datetime import datetime, timezone, timedelta
import sys
import pandas as pd
import psycopg2
import os

# ==============================
# Config
# ==============================
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'Betmobile',
    'user': 'postgres',
    'password': '300500'
}

WEEK_MODE_LEAGUES = {"Finland"}

TARGET_SEASON = os.getenv("TARGET_SEASON", "2025")

USE_ODDS_HISTORY = False
USE_ODDS_FEATURE_VIEWS = False

ODDS_BOOK_NAME   = 'oddspedia_presented'
ODDS_SOURCE_NAME = 'oddspedia_presented_bookmaker'

TARGET_TABLE = "oddspedia_unibet_backbone"

ACTIVE_LEAGUE_KEYS = [
    "Finland",
]

BACKFILL_PREVIOUS_WEEKS = int(os.getenv("BACKFILL_PREVIOUS_WEEKS", 4))
BACKFILL_START_DATE = os.getenv("BACKFILL_START_DATE", "")

# Cutoff / refresh (voor de MVs die odds-dynamiek bouwen)
USE_ODDS_FEATURE_VIEWS = True
CUTOFF_HOURS_BEFORE_KICKOFF = 6   # kickoff - 6 uur

# Rolling venster om te bepalen wat we bewaren/schrijven (nu ook via ENV override)
KEEP_PAST_DAYS   = int(os.getenv("KEEP_PAST_DAYS", 2))     # standaard 2 dagen terug
KEEP_FUTURE_DAYS = int(os.getenv("KEEP_FUTURE_DAYS", 21))  # standaard 21 dagen vooruit

# Centrale configuratie voor alle competities
LEAGUES_CONFIG = {
    # UEFA Competitions
    'Champions League': {'name': 'UEFA Champions League', 'url': 'nl/voetbal/europa/champions-league#odds'},
    'Europa League': {'name': 'UEFA Europa League', 'url': 'nl/voetbal/europa/europa-league#odds'},
    'Conference League': {'name': 'UEFA Europa Conference League', 'url': 'nl/voetbal/europa/europa-conference-league#odds'},
    # Major Leagues
    'England': {'name': 'England Premier League', 'url': 'nl/voetbal/engeland/premier-league#odds'},
    'Spain': {'name': 'Spain La Liga', 'url': 'nl/voetbal/spanje/la-liga#odds'},
    'Germany': {'name': 'Germany Bundesliga', 'url': 'nl/voetbal/duitsland/bundesliga#odds'},
    'Italy': {'name': 'Italy Serie A', 'url': 'nl/voetbal/italie/serie-a#odds'},
    'France': {'name': 'France Ligue 1', 'url': 'nl/voetbal/frankrijk/ligue-1#odds'},
    'Netherlands': {'name': 'Netherlands Eredivisie', 'url': 'nl/voetbal/nederland/eredivisie#odds'},
    'Portugal': {'name': 'Portugal Primeira Liga', 'url': 'nl/voetbal/portugal/liga-portugal-bwin#odds'},
    'Belgium': {'name': 'Belgium Pro League', 'url': 'nl/voetbal/belgie/eerste-klasse-a#odds'},
    # Mid-tier Leagues
    'Austria': {'name': 'Austria Bundesliga', 'url': 'nl/a/voetbal/oostenrijk/bundesliga#odds'},
    'Switzerland': {'name': 'Switzerland Super League', 'url': 'nl/a/voetbal/zwitserland/super-league#odds'},
    'Türkiye': {'name': 'Türkiye Super Lig', 'url': 'nl/a/voetbal/turkije/superlig#odds'},
    'Scotland': {'name': 'Scotland Premiership', 'url': 'nl/a/voetbal/schotland/premiership#odds'},
    'Greece': {'name': 'Greece Super League', 'url': 'nl/a/voetbal/griekenland/super-league#odds'},
    'Czech Republic': {'name': 'Czech Republic First League', 'url': 'nl/a/voetbal/tsjechie/first-division#odds'},
    'Denmark': {'name': 'Denmark Superliga', 'url': 'nl/a/voetbal/denemarken/superliga#odds'},
    'Norway': {'name': 'Norway Eliteserien', 'url': 'nl/a/voetbal/noorwegen/tippeligaen#odds'},
    'Sweden': {'name': 'Sweden Allsvenskan', 'url': 'nl/a/voetbal/zweden/allsvenskan#odds'},
    # Small Leagues
    'Albania': {'name': 'Albania Superliga', 'url': 'nl/a/voetbal/albanie/superliga#odds'},
    'Andorra': {'name': 'Andorra Primera Divisio', 'url': 'nl/a/voetbal/andorra/primera-division#odds'},
    'Armenia': {'name': 'Armenia Premier League', 'url': 'nl/a/voetbal/armenie/premier-league#odds'},
    'Azerbaijan': {'name': 'Azerbaijan Premier League', 'url': 'nl/a/voetbal/azerbeidzjan/premier-league#odds'},
    'Belarus': {'name': 'Belarus Premier League', 'url': 'nl/a/voetbal/belarus/premier-league#odds'},
    'Bosnia': {'name': 'Bosnia Premier League', 'url': 'nl/a/voetbal/bosnie-en-herzegovina/premier-liga#odds'},
    'Bulgaria': {'name': 'Bulgaria First League', 'url': 'nl/a/voetbal/bulgarije/first-professional-leag#odds'},
    'Croatia': {'name': 'Croatia 1. HNL', 'url': 'nl/a/voetbal/kroatie/hnl#odds'},
    'Cyprus': {'name': 'Cyprus First Division', 'url': 'nl/a/voetbal/cyprus/first-division#odds'},
    'Estonia': {'name': 'Estonia Meistriliiga', 'url': 'nl/a/voetbal/estland/meistriliiga#odds'},
    'Faroe Islands': {'name': 'Faroe Islands Premier League', 'url': 'nl/a/voetbal/faroe-islands/premier-league#odds'},
    'Finland': {'name': 'Finland Veikkausliiga', 'url': 'nl/a/voetbal/finland/veikkausliiga#odds'},
    'Georgia': {'name': 'Georgia Erovnuli Liga', 'url': 'nl/a/voetbal/georgie/national-league#odds'},
    'Gibraltar': {'name': 'Gibraltar National League', 'url': 'nl/a/voetbal/gibraltar/premier-league#odds'},
    'Hungary': {'name': 'Hungary NB I', 'url': 'nl/a/voetbal/hongarije/nbi-liga#odds'},
    'Iceland': {'name': 'Iceland Premier League', 'url': 'nl/a/voetbal/ijsland/efsta-deild#odds'},
    'Ireland': {'name': 'Ireland Premier Division', 'url': 'nl/a/voetbal/ierland/irish-premier-division#odds'},
    'Israel': {'name': 'Israel Premier League', 'url': 'nl/a/voetbal/israel/premier-league#odds'},
    'Kazakhstan': {'name': 'Kazakhstan Premier League', 'url': 'nl/a/voetbal/kazachstan/premier-league#odds'},
    'Kosovo': {'name': 'Kosovo Superliga', 'url': 'nl/a/voetbal/kosovo/superliga#odds'},
    'Latvia': {'name': 'Latvia Virsliga', 'url': 'nl/a/voetbal/letland/higher-league#odds'},
    'Lithuania': {'name': 'Lithuania A Lyga', 'url': 'nl/a/voetbal/litouwen/a-liga#odds'},
    'Luxembourg': {'name': 'Luxembourg National Division', 'url': 'nl/a/voetbal/luxemburg/national-division#odds'},
    'Malta': {'name': 'Malta Premier League', 'url': 'nl/a/voetbal/malta/premier-division#odds'},
    'Moldova': {'name': 'Moldova National Division', 'url': 'nl/a/voetbal/moldavie/national-division#odds'},
    'Montenegro': {'name': 'Montenegro First League', 'url': 'nl/a/voetbal/montenegro/first-league#odds'},
    'North Macedonia': {'name': 'North Macedonia First League', 'url': 'nl/a/voetbal/noord-macedonie/1-mfl#odds'},
    'Northern Ireland': {'name': 'Northern Ireland Premiership', 'url': 'nl/a/voetbal/noord-ierland/nifl-premiership#odds'},
    'Poland': {'name': 'Poland Ekstraklasa', 'url': 'nl/a/voetbal/polen/ekstraklasa#odds'},
    'Romania': {'name': 'Romania Liga 1', 'url': 'nl/a/voetbal/roemenie/liga-1#odds'},
    'Russia': {'name': 'Russia Premier League', 'url': 'nl/a/voetbal/rusland/premier-league#odds'},
    'San Marino': {'name': 'San Marino Campionato', 'url': 'nl/a/voetbal/san-marino/campionato-sammarinese#odds'},
    'Serbia': {'name': 'Serbia Super Liga', 'url': 'nl/a/voetbal/servie/super-liga#odds'},
    'Slovakia': {'name': 'Slovakia Super Liga', 'url': 'nl/a/voetbal/slowakije/super-liga#odds'},
    'Slovenia': {'name': 'Slovenia Prva Liga', 'url': 'nl/a/voetbal/slovenie/1-snl#odds'},
    'Ukraine': {'name': 'Ukraine Premier League', 'url': 'nl/a/voetbal/oekraine/premier-league#odds'},
    'Wales': {'name': 'Wales Premier League', 'url': 'nl/a/voetbal/wales/cymru-premier#odds'}
}

BASE_URL = "https://oddspedia.com/"
CURRENT_USER = "dekraan"

# ==============================
# Scraper
# ==============================

class OddspediaScraper:
    def __init__(self):
        self.current_date = datetime.now(timezone.utc)
        print(f"Start (UTC): {self.current_date:%Y-%m-%d %H:%M:%S}")
        self.ensure_core_tables()

        if USE_ODDS_HISTORY:
            self.ensure_odds_history_tables_and_views()

        self.leagues = [
            {"key": key, "name": cfg["name"], "url": cfg["url"]}
            for key, cfg in LEAGUES_CONFIG.items()
            if key in ACTIVE_LEAGUE_KEYS
        ]
        
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-geolocation")
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])

        if os.environ.get("HEADLESS") == "1":
            options.add_argument("--headless=new")
            options.add_argument("--window-size=1920,1080")

        USE_EXISTING_CHROME = True

        if USE_EXISTING_CHROME:
            print("Connecting to existing Chrome...")

            options = webdriver.ChromeOptions()
            options.debugger_address = "127.0.0.1:9222"

            self.driver = webdriver.Chrome(options=options)

        else:
            print("Launching Chrome…")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)

        self.wait = WebDriverWait(self.driver, 20)

        print("Browser ready.")

        # Counters
        self.skipped_out_of_window = 0
        self.skipped_history = 0

    # ---------- DB setup ----------
    def ensure_core_tables(self):
        """Create/alter oddspedia_data + create alias tables + trigger (idempotent)."""
        with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS public.oddspedia_unibet_backbone
            (
                match_id varchar PRIMARY KEY,
                date varchar,
                home_team varchar,
                away_team varchar,
                odds_home double precision,
                odds_draw double precision,
                odds_away double precision,
                score varchar,
                competition varchar,
                created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
                updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
                created_by varchar DEFAULT 'dekraan',
                CONSTRAINT oddspedia_unibet_backbone_unique_match UNIQUE (date, home_team, away_team)
            );""")
            cur.execute("""
            ALTER TABLE public.oddspedia_unibet_backbone
              ADD COLUMN IF NOT EXISTS status varchar(20) NOT NULL DEFAULT 'scheduled',
              ADD COLUMN IF NOT EXISTS status_raw text,
              ADD COLUMN IF NOT EXISTS status_updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
              ADD COLUMN IF NOT EXISTS source_url text,
              ADD COLUMN IF NOT EXISTS home_team_canon text,
              ADD COLUMN IF NOT EXISTS away_team_canon text,
              ADD COLUMN IF NOT EXISTS match_id_canon  text,
              ADD COLUMN IF NOT EXISTS needs_mapping   boolean NOT NULL DEFAULT false;""")
            cur.execute("""
            CREATE OR REPLACE FUNCTION public.trg_touch_status_timestamp()
            RETURNS trigger AS $$
            BEGIN
              IF NEW.status IS DISTINCT FROM OLD.status THEN
                NEW.status_updated_at := CURRENT_TIMESTAMP;
              END IF;
              RETURN NEW;
            END $$ LANGUAGE plpgsql;""")
            cur.execute("DROP TRIGGER IF EXISTS t_status_touch ON public.oddspedia_unibet_backbone;")
            cur.execute("""
            CREATE TRIGGER t_status_touch
            BEFORE UPDATE ON public.oddspedia_unibet_backbone
            FOR EACH ROW EXECUTE FUNCTION public.trg_touch_status_timestamp();""")

            # alias + pending
            cur.execute("""
            CREATE TABLE IF NOT EXISTS public.team_aliases (
              alias_name     text PRIMARY KEY,
              canonical_name text NOT NULL,
              source         text DEFAULT 'manual',
              league         text,
              active         boolean DEFAULT true,
              created_at     timestamptz DEFAULT now(),
              updated_at     timestamptz DEFAULT now()
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS public.pending_aliases (
              alias_name     text PRIMARY KEY,
              first_seen_at  timestamptz DEFAULT now(),
              last_seen_at   timestamptz DEFAULT now(),
              sample_league  text
            );""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_oddspedia_match_id_canon ON public.oddspedia_unibet_backbone(match_id_canon);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_oddspedia_canon_triplet ON public.oddspedia_unibet_backbone(date, home_team_canon, away_team_canon);")
            conn.commit()

    def ensure_odds_history_tables_and_views(self):
        """odds_history + alle materialized views (idempotent)."""
        if not USE_ODDS_HISTORY:
            return
        sql = f"""
        -- odds_history
        CREATE TABLE IF NOT EXISTS odds_history (
          match_id    text        NOT NULL,
          book        text        NOT NULL DEFAULT 'avg',
          scraped_at  timestamptz NOT NULL DEFAULT now(),
          odds_home   double precision,
          odds_draw   double precision,
          odds_away   double precision,
          source      text,
          PRIMARY KEY (match_id, book, scraped_at)
        );
        CREATE INDEX IF NOT EXISTS idx_odds_history_match_time ON odds_history (match_id, scraped_at);
        CREATE INDEX IF NOT EXISTS idx_odds_history_time       ON odds_history (scraped_at);

        -- CUTOFF-set
        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_latest_before_cutoff AS
        WITH m AS (
          SELECT match_id, (date::timestamptz) AS kickoff_at
          FROM betmobile
        ),
        snap AS (
          SELECT h.match_id, h.scraped_at, h.odds_home, h.odds_draw, h.odds_away,
                 m.kickoff_at, (m.kickoff_at - interval '{CUTOFF_HOURS_BEFORE_KICKOFF} hours') AS cutoff_at
          FROM odds_history h
          JOIN m ON m.match_id = h.match_id
          WHERE h.book = '{ODDS_BOOK_NAME}'
            AND h.scraped_at <= (m.kickoff_at - interval '{CUTOFF_HOURS_BEFORE_KICKOFF} hours')
        ),
        ranked AS (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY match_id ORDER BY scraped_at DESC) AS rn
          FROM snap
        )
        SELECT match_id, scraped_at, odds_home, odds_draw, odds_away, kickoff_at, cutoff_at
        FROM ranked
        WHERE rn = 1;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_latest_cutoff_match ON odds_latest_before_cutoff (match_id);

        -- First seen
        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_first_seen AS
        WITH ranked AS (
          SELECT h.match_id, h.scraped_at,
                 h.odds_home, h.odds_draw, h.odds_away,
                 ROW_NUMBER() OVER (PARTITION BY h.match_id ORDER BY h.scraped_at ASC) AS rn
          FROM odds_history h
          WHERE h.book = '{ODDS_BOOK_NAME}'
        )
        SELECT match_id,
               scraped_at AS first_seen_at,
               odds_home  AS home_open,
               odds_draw  AS draw_open,
               odds_away  AS away_open
        FROM ranked
        WHERE rn = 1;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_first_seen_match ON odds_first_seen (match_id);

        -- pre-cutoff aggregaat
        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_pre_cutoff_agg AS
        WITH cutoff AS (
          SELECT match_id, cutoff_at
          FROM odds_latest_before_cutoff
        ),
        hist AS (
          SELECT h.match_id, h.scraped_at, h.odds_home, h.odds_draw, h.odds_away, c.cutoff_at
          FROM odds_history h
          JOIN cutoff c USING (match_id)
          WHERE h.book = '{ODDS_BOOK_NAME}' AND h.scraped_at <= c.cutoff_at
        )
        SELECT
          match_id,
          MIN(scraped_at) AS first_seen_at,
          MAX(scraped_at) AS last_seen_at,
          COUNT(*)        AS n_snapshots,
          MIN(odds_home)  AS home_min, MAX(odds_home) AS home_max,
          MIN(odds_draw)  AS draw_min, MAX(odds_draw) AS draw_max,
          MIN(odds_away)  AS away_min, MAX(odds_away) AS away_max
        FROM hist
        GROUP BY match_id;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pre_cutoff_agg_match ON odds_pre_cutoff_agg (match_id);

        -- dynamiek t.o.v. cutoff
        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_dynamics_features AS
        SELECT
          m.match_id,
          l.scraped_at AS last_scraped_at,
          l.kickoff_at,
          l.cutoff_at,
          l.odds_home  AS home_last,
          l.odds_draw  AS draw_last,
          l.odds_away  AS away_last,
          f.first_seen_at,
          f.home_open,
          f.draw_open,
          f.away_open,
          (l.odds_home - f.home_open)                 AS home_drift_abs,
          (l.odds_home / NULLIF(f.home_open,0)) - 1.0 AS home_drift_pct,
          (l.odds_away - f.away_open)                 AS away_drift_abs,
          (l.odds_away / NULLIF(f.away_open,0)) - 1.0 AS away_drift_pct,
          a.n_snapshots,
          (a.home_max - a.home_min)                   AS home_range,
          (a.away_max - a.away_min)                   AS away_range,
          EXTRACT(EPOCH FROM (l.cutoff_at - l.scraped_at))/3600.0 AS hours_stale
        FROM odds_latest_before_cutoff l
        JOIN odds_first_seen      f USING (match_id)
        JOIN odds_pre_cutoff_agg  a USING (match_id)
        JOIN betmobile            m USING (match_id);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_dyn_features_match ON odds_dynamics_features (match_id);

        -- NU-set (geen cutoff)
        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_latest_until_now AS
        WITH ranked AS (
          SELECT
            match_id, scraped_at,
            odds_home, odds_draw, odds_away,
            ROW_NUMBER() OVER (PARTITION BY match_id ORDER BY scraped_at DESC) AS rn
          FROM odds_history
          WHERE book = '{ODDS_BOOK_NAME}'
        )
        SELECT
          match_id,
          scraped_at AS last_scraped_at,
          odds_home  AS home_last,
          odds_draw  AS draw_last,
          odds_away  AS away_last
        FROM ranked
        WHERE rn = 1;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_latest_now_match ON odds_latest_until_now (match_id);

        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_agg_until_now AS
        SELECT
          match_id,
          MIN(scraped_at) AS first_seen_at,
          MAX(scraped_at) AS last_seen_at,
          COUNT(*)        AS n_snapshots,
          MIN(odds_home)  AS home_min, MAX(odds_home) AS home_max,
          MIN(odds_draw)  AS draw_min, MAX(odds_draw) AS draw_max,
          MIN(odds_away)  AS away_min, MAX(odds_away) AS away_max
        FROM odds_history
        WHERE book = '{ODDS_BOOK_NAME}'
        GROUP BY match_id;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agg_now_match ON odds_agg_until_now (match_id);

        CREATE MATERIALIZED VIEW IF NOT EXISTS odds_dynamics_features_now AS
        SELECT
          l.match_id,
          l.last_scraped_at,
          f.first_seen_at,
          l.home_last, l.draw_last, l.away_last,
          f.home_open, f.draw_open, f.away_open,
          (l.home_last - f.home_open)                 AS home_drift_abs,
          (l.home_last / NULLIF(f.home_open,0)) - 1.0 AS home_drift_pct,
          (l.away_last - f.away_open)                 AS away_drift_abs,
          (l.away_last / NULLIF(f.away_open,0)) - 1.0 AS away_drift_pct,
          a.n_snapshots,
          (a.home_max - a.home_min)                   AS home_range,
          (a.away_max - a.away_min)                   AS away_range
        FROM odds_latest_until_now l
        JOIN odds_first_seen     f USING (match_id)
        JOIN odds_agg_until_now  a USING (match_id);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_dyn_now_match ON odds_dynamics_features_now (match_id);
        """
        with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()

    def refresh_odds_feature_views(self):
        if not (USE_ODDS_HISTORY and USE_ODDS_FEATURE_VIEWS):
            return
        with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            for mv in [
                "odds_latest_before_cutoff",
                "odds_first_seen",
                "odds_pre_cutoff_agg",
                "odds_dynamics_features",
                "odds_latest_until_now",
                "odds_agg_until_now",
                "odds_dynamics_features_now",
            ]:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv};")
            conn.commit()

    # ---------- utilities ----------
    def in_window(self, date_str):
        """Check of YYYY-MM-DD in [now-KEEP_PAST_DAYS, now+KEEP_FUTURE_DAYS] valt."""
        if not date_str:
            return False
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return False
        start = self.current_date - timedelta(days=KEEP_PAST_DAYS)
        end   = self.current_date + timedelta(days=KEEP_FUTURE_DAYS)
        return start.date() <= d.date() <= end.date()

    def clean_odds(self, val):
        if pd.isna(val) or val is None or val == "":
            return None
        try:
            return round(float(val), 3)
        except Exception:
            return None

    def convert_date_headline(self, date_str):
        """Oddspedia headline → YYYY-MM-DD, gebaseerd op TARGET_SEASON."""
        try:
            cleaned = date_str.split('-')[0].strip()
            parts = cleaned.split()

            day = int(parts[0])
            mon = parts[1][:3].upper()

            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }

            month = month_map[mon]

            if "/" in TARGET_SEASON:
                start_yy, end_yy = TARGET_SEASON.split("/")
                start_year = 2000 + int(start_yy)
                end_year = 2000 + int(end_yy)

                year = start_year if month >= 8 else end_year
            else:
                year = int(TARGET_SEASON)

            return datetime(year, month, day).strftime("%Y-%m-%d")

        except Exception:
            return None

    def select_season(self, season_label: str):
        try:
            print(f"Selecting season: {season_label}")

            toggle = self.wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR,
                ".content__header--league__dropdown .old-dropdown__toggle"
            )))

            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", toggle)
            time.sleep(0.3)
            self.driver.execute_script("arguments[0].click();", toggle)
            time.sleep(1)

            # Zoek zichtbaar dropdown-item met exact de tekst, bv 25/26
            option = self.wait.until(EC.presence_of_element_located((
                By.XPATH,
                f"//*[contains(@class,'old-dropdown') and normalize-space()='{season_label}']"
            )))

            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", option)
            time.sleep(0.3)
            self.driver.execute_script("arguments[0].click();", option)

            print(f"Clicked season: {season_label}")
            time.sleep(6)

            print(f"Current marker: {self._week_marker()}")
            print(f"Current URL: {self.driver.current_url}")

        except Exception as e:
            print(f"(warn) select season {season_label}: {e}")

    def select_round_mode(self):
        """
        Zet Oddspedia-weergave op Round i.p.v. Week.
        Handig voor historische seizoenen: dan kun je ronde voor ronde navigeren.
        """
        try:
            print("Selecting Round mode…")

            before_marker = self._week_marker()

            round_radio = self.wait.until(EC.element_to_be_clickable((
                By.XPATH,
                "//label[contains(@class,'old-radio__inner')][.//input[@value='round']]"
            )))

            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", round_radio)
            time.sleep(0.2)
            self.driver.execute_script("arguments[0].click();", round_radio)

            changed = self._wait_for_week_change(before_marker, timeout=10)

            if not changed:
                time.sleep(3)

            print("Round mode selected.")
            print(f"Current marker: {self._week_marker()}")

        except Exception as e:
            print(f"(warn) select round mode: {e}")

    def open_earlier_results(self, max_tries=2):
        for _ in range(max_tries):
            try:
                before = len(self.driver.find_elements(By.CSS_SELECTOR, ".match-list-item, .game-list-item"))
                btns = self.driver.find_elements(By.CSS_SELECTOR, "button.show-earlier-results-button, .show-earlier-results-button")
                btns = [b for b in btns if b.is_displayed() and b.is_enabled()]
                if not btns:
                    return False
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btns[0])
                time.sleep(0.2)
                self.driver.execute_script("arguments[0].click();", btns[0])
                end = time.time() + 6
                while time.time() < end:
                    time.sleep(0.3)
                    after  = len(self.driver.find_elements(By.CSS_SELECTOR, ".match-list-item, .game-list-item"))
                    still = any(b.is_displayed() for b in self.driver.find_elements(By.CSS_SELECTOR, "button.show-earlier-results-button, .show-earlier-results-button"))
                    if after > before or not still:
                        return True
            except Exception:
                continue
        return False

    def _week_marker(self):
        """
        Unieke marker voor de getoonde week (headline of eerste match).
        Werkt voor zowel oude (.match-*) als nieuwe (.game-*) DOM.
        """
        # nieuwe headline
        try:
            heads = self.driver.find_elements(By.CSS_SELECTOR, ".game-list-headline-league")
            for h in heads:
                txt = (h.text or "").strip()
                if txt:
                    return txt.split("\n", 1)[0].strip()
        except Exception:
            pass

        # oude headline
        try:
            heads = self.driver.find_elements(By.CSS_SELECTOR, ".match-list-headline-league")
            for h in heads:
                txt = (h.text or "").strip()
                if txt:
                    return txt.split("\n", 1)[0].strip()
        except Exception:
            pass

        # eerste item (nieuw/oud)
        try:
            first = self.driver.find_elements(By.CSS_SELECTOR, ".match-list-item, .game-list-item")[:1]
            if first:
                t = first[0].find_elements(By.CSS_SELECTOR, ".team-names-stack span.text-truncate, .match-team__name")
                tvals = [x.text.strip() for x in t if (x.text or "").strip()]
                if len(tvals) >= 2:
                    return f"{tvals[0]} vs {tvals[1]}"
        except Exception:
            pass

        return f"marker-{int(time.time()*1000)}"


    def _wait_for_week_change(self, before_marker, timeout=12):
        """
        Wacht totdat de marker anders is dan before_marker.
        """
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(0.3)
            now = self._week_marker()
            if now and now != before_marker:
                return True
        return False

    def navigate_week(self, direction='previous'):
        """
        Klik 'vorige' of 'volgende' week en WACHT tot de lijst echt vernieuwd is.
        Probeert: normale click -> JS click -> tweede poging.
        """
        try:
            sel = "button.ml-pagination__btn--prev" if direction == 'previous' else "button.ml-pagination__btn--next"
            # zorg dat de knop in beeld is
            self.driver.execute_script("window.scrollTo(0, 0);")
            btn = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))

            before_marker = self._week_marker()

            # 1) normale click
            try:
                btn.click()
            except Exception:
                # fallback: JS click
                self.driver.execute_script("arguments[0].click();", btn)

            if self._wait_for_week_change(before_marker, timeout=12):
                return

            # 2) tweede poging: her-vind + JS click
            time.sleep(0.5)
            self.driver.execute_script("window.scrollTo(0, 0);")
            btn2 = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
            self.driver.execute_script("arguments[0].click();", btn2)

            if not self._wait_for_week_change(before_marker, timeout=12):
                print(f"(warn) navigate {direction}: content marker did not change; likely same week still shown.")
        except Exception as e:
            print(f"(warn) navigate {direction}: {e}")


    # ---------- extractors ----------
    def extract_score_text(self, item):
        try:
            gs = item.find_elements(By.CSS_SELECTOR, ".game-score-result span")
            if len(gs) >= 2:
                h = (gs[0].text or "").strip()
                a = (gs[1].text or "").strip()
                if h.isdigit() and a.isdigit():
                    return f"{h}-{a}"
        except Exception:
            pass
        try:
            scores = item.find_elements(By.CSS_SELECTOR, ".old-match-score .old-match-score-result__score")
            if len(scores) >= 2:
                h = scores[0].get_attribute("textContent").strip()
                a = scores[1].get_attribute("textContent").strip()
                if h.isdigit() and a.isdigit():
                    return f"{h}-{a}"
        except Exception:
            pass
        selectors = [
            (".match-score__team--home .match-score-result__score", ".match-score__team--away .match-score-result__score"),
            (".match-score-result__home", ".match-score-result__away"),
            (".match-score .score__home", ".match-score .score__away"),
            (".match-team__score--home", ".match-team__score--away"),
        ]
        for sh, sa in selectors:
            try:
                h = item.find_element(By.CSS_SELECTOR, sh).get_attribute("textContent").strip()
                a = item.find_element(By.CSS_SELECTOR, sa).get_attribute("textContent").strip()
                if h and a and h.isdigit() and a.isdigit():
                    return f"{h}-{a}"
            except Exception:
                continue
        return None

    def extract_status_text(self, item):
        """Return (status_norm, status_raw)."""
        try:
            el = item.find_element(By.CSS_SELECTOR, ".match-status--special")
            raw = el.get_attribute("textContent").strip().lower()
            if "postponed" in raw or "pp" in raw:
                return "postponed", raw
            if "canceled" in raw or "cancelled" in raw:
                return "canceled", raw
            if "abandoned" in raw:
                return "abandoned", raw
            return "scheduled", raw
        except Exception:
            pass
        try:
            s = item.find_element(By.CSS_SELECTOR, ".match-date__status")
            raw = (s.get_attribute("textContent") or "").strip().lower() or (s.get_attribute("title") or "")
            if "postponed" in raw:
                return "postponed", raw
            if "cancelled" in raw or "canceled" in raw:
                return "canceled", raw
            if "abandoned" in raw:
                return "abandoned", raw
        except Exception:
            pass
        try:
            el = item.find_element(By.CSS_SELECTOR, ".match-status--inplay")
            raw = el.get_attribute("textContent").strip()
            return "live", raw or "Inplay"
        except Exception:
            pass
        try:
            el = item.find_element(By.CSS_SELECTOR, ".match-status")
            raw = el.get_attribute("textContent").strip()
            up = raw.upper()
            if "FT" in up:
                return "finished", raw
            if any(k in up for k in ("HT", "ET", "PEN", "LIVE", "INPLAY")):
                return "live", raw
            return "scheduled", raw
        except Exception:
            pass
        try:
            el = item.find_element(By.CSS_SELECTOR, ".match-date__time")
            raw = el.get_attribute("textContent").strip()
            if raw:
                return "scheduled", raw
        except Exception:
            pass
        return "scheduled", None

    def extract_match_url(self, item):
        try:
            a = item.find_element(By.CSS_SELECTOR, "a.match-url")
            href = a.get_attribute("href")
            if href:
                return href if href.startswith("http") else BASE_URL.rstrip("/") + href
        except Exception:
            pass
        try:
            a = item.find_element(By.CSS_SELECTOR, "a.match-url--flex")
            href = a.get_attribute("href")
            if href:
                return href if href.startswith("http") else BASE_URL.rstrip("/") + href
        except Exception:
            pass
        return None

    # ---------- canonical helpers ----------
    def update_canonical_fields_and_pending(self, cur, match_id, date_str, home_team_raw, away_team_raw, competition):
        """
        - Zet home_team_canon/away_team_canon/match_id_canon/needs_mapping in oddspedia_data voor deze rij
        - Log onbekende namen in pending_aliases
        """
        # 1) Update canon-kolommen met join op team_aliases
        cur.execute("""
            WITH ah AS (
              SELECT canonical_name FROM team_aliases WHERE alias_name = %s AND active
            ),
            aa AS (
              SELECT canonical_name FROM team_aliases WHERE alias_name = %s AND active
            )
            UPDATE public.oddspedia_unibet_backbone o
               SET home_team_canon = COALESCE((SELECT canonical_name FROM ah), %s),
                   away_team_canon = COALESCE((SELECT canonical_name FROM aa), %s),
                   match_id_canon  = %s || '_' ||
                                     COALESCE((SELECT canonical_name FROM ah), %s) || '_' ||
                                     COALESCE((SELECT canonical_name FROM aa), %s),
                   needs_mapping   = ((SELECT canonical_name FROM ah) IS NULL OR (SELECT canonical_name FROM aa) IS NULL)
             WHERE o.match_id = %s;
        """, (
            home_team_raw, away_team_raw,
            home_team_raw, away_team_raw,
            date_str, home_team_raw, away_team_raw,
            match_id
        ))

        # 2) Pending loggen als onbekend
        cur.execute("""
            INSERT INTO pending_aliases (alias_name, first_seen_at, last_seen_at, sample_league)
            SELECT %s, now(), now(), %s
            WHERE NOT EXISTS (SELECT 1 FROM team_aliases WHERE alias_name = %s)
            ON CONFLICT (alias_name) DO UPDATE
              SET last_seen_at = EXCLUDED.last_seen_at;
        """, (home_team_raw, competition, home_team_raw))
        cur.execute("""
            INSERT INTO pending_aliases (alias_name, first_seen_at, last_seen_at, sample_league)
            SELECT %s, now(), now(), %s
            WHERE NOT EXISTS (SELECT 1 FROM team_aliases WHERE alias_name = %s)
            ON CONFLICT (alias_name) DO UPDATE
              SET last_seen_at = EXCLUDED.last_seen_at;
        """, (away_team_raw, competition, away_team_raw))

    # ---------- persist ----------
    def save_odds_snapshot_row(self, cur, match_id, odds_home, odds_draw, odds_away,
                               match_date_str, status, score):
        """
        Sla één odds-snapshot op zolang de match nog niet klaar is.
        - Geen vensterfilter: we vertrouwen op je workflow (current + previous week).
        - Skip zodra status 'finished' is of er al een score staat.
        """
        if not USE_ODDS_HISTORY:
            return
        if odds_home is None and odds_draw is None and odds_away is None:
            return

        # STOP snapshots zodra er FT of een score is
        if (status and status.lower() == 'finished') or (score not in (None, '', 'NULL')):
            self.skipped_history += 1
            return

        cur.execute("""
            INSERT INTO odds_history (match_id, book, odds_home, odds_draw, odds_away, source)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (match_id, ODDS_BOOK_NAME, odds_home, odds_draw, odds_away, ODDS_SOURCE_NAME))

    def save_batch(self, matches, competition):
        if not matches:
            return
        with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            for m in matches:
                # geen window-guard meer:
                pass

                if not m.get("date"):
                    print(f"(skip) geen datum: {competition} | {m.get('home_team')} - {m.get('away_team')}")
                    continue

                match_id = f"{m['date']}_{m['home_team']}_{m['away_team']}"
                odds_home = self.clean_odds(m['odds']['home'])
                odds_draw = self.clean_odds(m['odds']['draw'])
                odds_away = self.clean_odds(m['odds']['away'])

                # Raw upsert
                cur.execute("""
                    INSERT INTO public.oddspedia_unibet_backbone (
                        match_id, date, home_team, away_team,
                        odds_home, odds_draw, odds_away,
                        score, competition, created_at, created_by,
                        status, status_raw, source_url
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )
                    ON CONFLICT ON CONSTRAINT oddspedia_unibet_backbone_unique_match
                    DO UPDATE SET
                        odds_home = CASE
                                      WHEN public.oddspedia_unibet_backbone.status = 'finished'
                                        THEN COALESCE(public.oddspedia_unibet_backbone.odds_home, EXCLUDED.odds_home)
                                      ELSE EXCLUDED.odds_home
                                    END,
                        odds_draw = CASE
                                      WHEN public.oddspedia_unibet_backbone.status = 'finished'
                                        THEN COALESCE(public.oddspedia_unibet_backbone.odds_draw, EXCLUDED.odds_draw)
                                      ELSE EXCLUDED.odds_draw
                                    END,
                        odds_away = CASE
                                      WHEN public.oddspedia_unibet_backbone.status = 'finished'
                                        THEN COALESCE(public.oddspedia_unibet_backbone.odds_away, EXCLUDED.odds_away)
                                      ELSE EXCLUDED.odds_away
                                    END,
                        score      = COALESCE(EXCLUDED.score, public.oddspedia_unibet_backbone.score),
                        status     = CASE
                            WHEN public.oddspedia_unibet_backbone.status = 'finished' THEN 'finished'
                            WHEN EXCLUDED.status = 'finished' THEN 'finished'
                            WHEN public.oddspedia_unibet_backbone.status = 'postponed' AND EXCLUDED.status = 'live' THEN 'live'
                            WHEN EXCLUDED.status IN ('postponed','canceled','abandoned')
                                 AND public.oddspedia_unibet_backbone.status <> 'finished' THEN EXCLUDED.status
                            WHEN EXCLUDED.status = 'live' AND public.oddspedia_unibet_backbone.status = 'scheduled' THEN 'live'
                            ELSE COALESCE(public.oddspedia_unibet_backbone.status, EXCLUDED.status)
                        END,
                        status_raw = COALESCE(EXCLUDED.status_raw, public.oddspedia_unibet_backbone.status_raw),
                        source_url = COALESCE(public.oddspedia_unibet_backbone.source_url, EXCLUDED.source_url),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    match_id, m['date'], m['home_team'], m['away_team'],
                    odds_home, odds_draw, odds_away,
                    m['score'], competition, self.current_date, CURRENT_USER,
                    m['status'], m['status_raw'], m['source_url']
                ))

                # Canonical mappen doen we straks later apart
                pass

                # odds_history snapshot (binnen venster + niet finished)
                self.save_odds_snapshot_row(cur, match_id, odds_home, odds_draw, odds_away,
                            m['date'], m['status'], m['score'])

            conn.commit()

    def parse_item_date(self, item):
        """
        Nieuwe Oddspedia DOM: datum staat bv als '04 Nov 25'
        of soms zonder jaar. Geeft 'YYYY-MM-DD' terug of None.
        """
        try:
            el = item.find_element(By.CSS_SELECTOR, ".match-date__time span")
            raw = (el.text or "").strip()

            if not raw:
                return None

            parts = raw.replace(",", " ").split()

            day = int(parts[0])
            mon = parts[1][:3].upper()

            mon_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }

            month = mon_map.get(mon)

            if not month:
                return None

            # Jaar aanwezig in de tekst, bv '04 Nov 25'
            if len(parts) >= 3:
                yy = parts[2]

                if len(yy) == 2 and yy.isdigit():
                    y = int(yy)
                    year = 2000 + y if y <= 79 else 1900 + y
                else:
                    year = int(yy)

            # Geen jaar aanwezig: afleiden uit TARGET_SEASON
            else:
                if "/" in TARGET_SEASON:
                    start_yy, end_yy = TARGET_SEASON.split("/")
                    start_year = 2000 + int(start_yy)
                    end_year = 2000 + int(end_yy)

                    year = start_year if month >= 8 else end_year
                else:
                    year = int(TARGET_SEASON)

            return datetime(year, month, day).strftime("%Y-%m-%d")

        except Exception:
            return None

    def parse_headline_date(self, text):
        """
        Parseert Oddspedia-kop zoals:
        '04 nov. dinsdag - Round 4'
        '21 okt. dinsdag - Round 3'

        Geeft YYYY-MM-DD terug.
        """
        try:
            if not text:
                return None

            raw = " ".join(str(text).lower().replace("\n", " ").split())

            month_map = {
                "jan": 1, "jan.": 1,
                "feb": 2, "feb.": 2,
                "mrt": 3, "mrt.": 3,
                "apr": 4, "apr.": 4,
                "mei": 5,
                "jun": 6, "jun.": 6,
                "jul": 7, "jul.": 7,
                "aug": 8, "aug.": 8,
                "sep": 9, "sep.": 9,
                "okt": 10, "okt.": 10,
                "nov": 11, "nov.": 11,
                "dec": 12, "dec.": 12,
            }

            parts = raw.split()
            day = int(parts[0])
            month = month_map.get(parts[1])

            if not month:
                return None

            # seizoen afleiden uit TARGET_SEASON
            if "/" in TARGET_SEASON:
                start_yy, end_yy = TARGET_SEASON.split("/")
                start_year = 2000 + int(start_yy)
                end_year = 2000 + int(end_yy)
                year = start_year if month >= 7 else end_year
            else:
                year = int(TARGET_SEASON)

            return datetime(year, month, day).strftime("%Y-%m-%d")

        except Exception:
            return None

    # ---------- parser ----------
    def parse_matches(self, league_name):
        """
        Parseert zowel oude (.match-list-*) als nieuwe (.game-list-*) Oddspedia DOM.
        """
        matches = []
        current_date = None  # fallback via headlines

        items = self.driver.find_elements(
            By.CSS_SELECTOR,
            ".match-list-item, .game-list-item, .match-list-headline-league, .game-list-headline-league"
        )

        for item in items:
            try:
                classes = item.get_attribute("class") or ""

                # --- Headline (datum / ronde) ---
                if "match-list-headline-league" in classes or "game-list-headline-league" in classes:
                    head_text = (item.text or "").strip()
                    parsed_headline_date = self.parse_headline_date(head_text)

                    if parsed_headline_date:
                        current_date = parsed_headline_date
                        print(f"Headline date: {current_date} | {head_text.splitlines()[0].strip()}")

                    continue

                # --- Basis record ---
                data = {
                    "league": league_name,
                    "date": current_date,  # fallback; overschrijven met item-datum indien beschikbaar
                    "home_team": None, "away_team": None,
                    "odds": {"home": None, "draw": None, "away": None},
                    "score": None,
                    "status": "scheduled",
                    "status_raw": None,
                    "source_url": None
                }

                # --- Datum (nieuwe DOM per item) ---
                item_date = self.parse_item_date(item)
                if item_date:
                    data["date"] = item_date

                # --- Teams: eerst nieuwe DOM, dan oude als fallback ---
                try:
                    tnodes = item.find_elements(By.CSS_SELECTOR, ".team-names-stack span.text-truncate")
                    tvals = [t.text.strip() for t in tnodes if (t.text or "").strip()]
                    if len(tvals) >= 2:
                        data["home_team"], data["away_team"] = tvals[:2]
                except Exception:
                    pass

                if not data["home_team"] or not data["away_team"]:
                    try:
                        teams = item.find_elements(By.CSS_SELECTOR, ".match-team__name")
                        if len(teams) >= 2:
                            data["home_team"] = teams[0].text.strip()
                            data["away_team"] = teams[1].text.strip()
                    except Exception:
                        pass

                # zonder teams heeft dit item geen zin
                if not data["home_team"] or not data["away_team"]:
                    continue

                # --- Status + Score ---
                status_norm, status_raw = self.extract_status_text(item)
                data["status"] = status_norm
                data["status_raw"] = status_raw

                score = self.extract_score_text(item)
                if score:
                    data["score"] = score
                    if data["status"] not in ("live", "postponed"):
                        data["status"] = "finished"
                        if not data["status_raw"]:
                            data["status_raw"] = "FT (inferred from score)"

                # --- Odds: eerst nieuwe DOM, dan oude fallback ---
                try:
                    odds_elems = item.find_elements(By.CSS_SELECTOR, ".match-odds .odd-box__value")
                    if len(odds_elems) >= 3:
                        data["odds"]["home"] = odds_elems[0].text.strip()
                        data["odds"]["draw"] = odds_elems[1].text.strip()
                        data["odds"]["away"] = odds_elems[2].text.strip()
                    if not data["odds"]["home"] or not data["odds"]["draw"] or not data["odds"]["away"]:
                        odds_elems = item.find_elements(By.CSS_SELECTOR, ".odd-box-with-logo__value, .odd__value")
                        if len(odds_elems) >= 3:
                            if not data["odds"]["home"]:
                                data["odds"]["home"] = odds_elems[0].text.strip()
                            if not data["odds"]["draw"]:
                                data["odds"]["draw"] = odds_elems[1].text.strip()
                            if not data["odds"]["away"]:
                                data["odds"]["away"] = odds_elems[2].text.strip()
                except Exception:
                    pass

                # --- Match URL ---
                data["source_url"] = self.extract_match_url(item)

                matches.append(data)

            except Exception as e:
                print(f"(warn) parse error: {e}")
                continue

        return matches


    # ---------- workflow ----------
    def run(self):
        try:
            for league in self.leagues:
                url = BASE_URL + league["url"]
                print(f"\nAccessing {league['name']}: {url}")
                self.driver.get(url)
                print("Waiting for page…")
                time.sleep(5)

                api_url = (
                    "https://oddspedia.com/api/v1/getMaxOddsWithPagination"
                    "?geoCode=NL"
                    "&bookmakerGeoCode=NL"
                    "&bookmakerGeoState="
                    "&wettsteuer=0"
                    "&startDate=2025-08-03T22%3A00%3A00Z"
                    "&endDate=2025-08-10T21%3A59%3A00Z"
                    "&sport=football"
                    "&category=netherlands"
                    "&league=eredivisie"
                    "&ot=100"
                    "&excludeSpecialStatus=0"
                    "&popularLeaguesOnly=0"
                    "&sortBy=default"
                    "&status=all"
                    "&page=1"
                    "&perPage=100"
                    "&seasonId=130943"
                    "&inplay=0"
                    "&language=en"
                )

                result = self.driver.execute_async_script("""
                    const url = arguments[0];
                    const done = arguments[1];

                    fetch(url, {
                        method: 'GET',
                        credentials: 'include',
                        headers: {
                            'accept': 'application/json, text/plain, */*'
                        }
                    })
                    .then(async response => {
                        const text = await response.text();
                        done({
                            status: response.status,
                            ok: response.ok,
                            text: text.slice(0, 2000)
                        });
                    })
                    .catch(error => {
                        done({
                            status: 0,
                            ok: false,
                            text: String(error)
                        });
                    });
                """, api_url)

                self.select_season(TARGET_SEASON)
                time.sleep(2)

                if league["key"] in WEEK_MODE_LEAGUES:
                    print("Finland/week-mode: Round mode overslaan.")
                else:
                    self.select_round_mode()
                    time.sleep(2)

                # Bookmakerselectie overslaan.
                # We gebruiken de standaard "presented by ..." odds van Oddspedia.
                time.sleep(1)

                if league["key"] == "Finland":
                    print("Finland mode: Week-weergave gebruiken en vooruit klikken.")

                    while True:
                        try:
                            while self.open_earlier_results():
                                time.sleep(0.2)
                        except Exception:
                            pass

                        time.sleep(0.3)

                        matches = self.parse_matches(league["name"])
                        self.save_batch(matches, league["name"])

                        before_marker = self._week_marker()
                        print(f"[{league['name']}] Saved week: {len(matches)} | marker: {before_marker}")

                        self.navigate_week("next")
                        time.sleep(2)

                        after_marker = self._week_marker()
                        print(f"Moved: {before_marker} -> {after_marker}")

                        if after_marker == before_marker:
                            print("No more weeks available.")
                            break

                else:
                    try:
                        while self.open_earlier_results():
                            time.sleep(0.2)
                    except Exception:
                        pass

                    time.sleep(0.3)

                    cur_matches = self.parse_matches(league["name"])
                    self.save_batch(cur_matches, league["name"])
                    print(f"[{league['name']}] Saved current round: {len(cur_matches)}")

                    while True:
                        before_marker = self._week_marker()

                        self.navigate_week("previous")
                        time.sleep(2)

                        after_marker = self._week_marker()

                        print(f"Moved: {before_marker} -> {after_marker}")

                        if after_marker == before_marker:
                            print("No more rounds available.")
                            break

                        try:
                            while self.open_earlier_results():
                                time.sleep(0.2)
                        except Exception:
                            pass

                        matches = self.parse_matches(league["name"])
                        self.save_batch(matches, league["name"])

                        print(f"Saved {len(matches)} matches")

            print("\nAll leagues processed. Done.")
        finally:
            try:
                self.driver.quit()
                print("Browser closed.")
            except Exception:
                pass

        print(f"\nSkipped (out of window): {self.skipped_out_of_window}")
        print(f"Skipped odds_history (finished/out-of-window): {self.skipped_history}")

        if USE_ODDS_HISTORY:
            print("\nRefreshing odds feature views…")
            self.refresh_odds_feature_views()
            print("Views refreshed. Done.")

# ==============================
# Entrypoint
# ==============================
if __name__ == "__main__":
    print(f"Starting scraper at {datetime.now(timezone.utc)}")
    try:
        OddspediaScraper().run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
