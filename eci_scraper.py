import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re
from datetime import datetime, timezone
import psycopg2
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
import random
import os
os.environ['WDM_LOG_LEVEL'] = '0' # Zet WebDriver Manager op stil

from betmobile_settings import DB_CONFIG

# Setup logging
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"eci_scraper_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Centrale configuratie voor alle competities blijft hetzelfde
LEAGUES_CONFIG = {
    # UEFA Competitions
    'Champions League': {'name': 'UEFA Champions League', 'id': '41'},
    'Europa League': {'name': 'UEFA Europa League', 'id': '43'},
    'Conference League': {'name': 'UEFA Europa Conference League', 'id': '1021'},
    
    # Major Leagues
    'England': {'name': 'England Premier League', 'id': '52'},
    'Spain': {'name': 'Spain La Liga', 'id': '67'},
    'Germany': {'name': 'Germany Bundesliga', 'id': '56'},
    'Italy': {'name': 'Italy Serie A', 'id': '53'},
    'France': {'name': 'France Ligue 1', 'id': '54'},
    'Netherlands': {'name': 'Netherlands Eredivisie', 'id': '2'},
    'Portugal': {'name': 'Portugal Primeira Liga', 'id': '69'},
    'Belgium': {'name': 'Belgium Pro League', 'id': '48'},
    
    # Mid-tier Leagues
    'Austria': {'name': 'Austria Bundesliga', 'id': '74'},
    'Switzerland': {'name': 'Switzerland Super League', 'id': '80'},
    'Türkiye': {'name': 'Türkiye Super Lig', 'id': '57'},
    'Scotland': {'name': 'Scotland Premiership', 'id': '55'},
    'Greece': {'name': 'Greece Super League', 'id': '96'},
    'Czech Republic': {'name': 'Czech Republic First League', 'id': '89'},
    'Denmark': {'name': 'Denmark Superliga', 'id': '71'},
    'Norway': {'name': 'Norway Eliteserien', 'id': '72'},
    'Sweden': {'name': 'Sweden Allsvenskan', 'id': '79'},
    
    # Small Leagues
    'Albania': {'name': 'Albania Superliga', 'id': '143'},
    'Andorra': {'name': 'Andorra Primera Divisio', 'id': '243'},
    'Armenia': {'name': 'Armenia Premier League', 'id': '249'},
    'Azerbaijan': {'name': 'Azerbaijan Premier League', 'id': '251'},
    'Belarus': {'name': 'Belarus Premier League', 'id': '250'},
    'Bosnia': {'name': 'Bosnia Premier League', 'id': '144'},
    'Bulgaria': {'name': 'Bulgaria First League', 'id': '90'},
    'Croatia': {'name': 'Croatia 1. HNL', 'id': '94'},
    'Cyprus': {'name': 'Cyprus First Division', 'id': '145'},
    'Estonia': {'name': 'Estonia Meistriliiga', 'id': '100'},
    'Faroe Islands': {'name': 'Faroe Islands Premier League', 'id': '242'},
    'Finland': {'name': 'Finland Veikkausliiga', 'id': '87'},
    'Georgia': {'name': 'Georgia Erovnuli Liga', 'id': '202'},
    'Gibraltar': {'name': 'Gibraltar National League', 'id': '892'},
    'Hungary': {'name': 'Hungary NB I', 'id': '141'},
    'Iceland': {'name': 'Iceland Premier League', 'id': '98'},
    'Ireland': {'name': 'Ireland Premier Division', 'id': '101'},
    'Israel': {'name': 'Israel Premier League', 'id': '240'},
    'Kazakhstan': {'name': 'Kazakhstan Premier League', 'id': '501'},
    'Kosovo': {'name': 'Kosovo Superliga', 'id': '922'},
    'Latvia': {'name': 'Latvia Virsliga', 'id': '252'},
    'Lithuania': {'name': 'Lithuania A Lyga', 'id': '146'},
    'Luxembourg': {'name': 'Luxembourg National Division', 'id': '241'},
    'Malta': {'name': 'Malta Premier League', 'id': '148'},
    'Moldova': {'name': 'Moldova National Division', 'id': '149'},
    'Montenegro': {'name': 'Montenegro First League', 'id': '656'},
    'North Macedonia': {'name': 'North Macedonia First League', 'id': '147'},
    'Northern Ireland': {'name': 'Northern Ireland Premiership', 'id': '150'},
    'Poland': {'name': 'Poland Ekstraklasa', 'id': '93'},
    'Romania': {'name': 'Romania Liga 1', 'id': '91'},
    'Russia': {'name': 'Russia Premier League', 'id': '76'},
    'San Marino': {'name': 'San Marino Campionato', 'id': '77'},
    'Serbia': {'name': 'Serbia Super Liga', 'id': '95'},
    'Slovakia': {'name': 'Slovakia Super Liga', 'id': '97'},
    'Slovenia': {'name': 'Slovenia Prva Liga', 'id': '151'},
    'Ukraine': {'name': 'Ukraine Premier League', 'id': '73'},
    'Wales': {'name': 'Wales Premier League', 'id': '173'}
}

def normalize_score(score):
    """
    Normaliseert een scorestring:
    - zet '1 - 0' -> '1-0'
    - accepteert lichte varianten
    - geeft None terug als er geen geldige score is
    """
    if not score:
        return None
    s = str(score).strip()
    # spaties rondom '-' weghalen
    import re as _re
    s = _re.sub(r'\s*-\s*', '-', s)
    # simpele 'd-d' case
    if _re.fullmatch(r'\d+-\d+', s):
        return s
    # fallback: pak de eerste twee integers in de string
    m = _re.search(r'(\d+)\D+(\d+)', s)
    return f"{m.group(1)}-{m.group(2)}" if m else None


class ECIScraper:
    def __init__(self):
        self.url = 'https://www.euroclubindex.com/match-odds/'
        self.current_timestamp = datetime.now(timezone.utc)
        logger.info(f"Current Date and Time (UTC): {self.current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Current User's Login: dekraan")
        
        # Database tabel aanmaken als die niet bestaat
        self.create_table()

    def parse_date(self, raw_date):
        """
        Parse dates like '6 Sept' (and variants) to 'YYYY-MM-DD'.
        Adds current year if missing, fixes non-standard month abbreviations,
        and falls back to dateutil for odd cases.
        Returns None if it really can't parse.
        """
        try:
            s = str(raw_date).strip()

            # Verwijder eventuele weekday aan het begin (Mon/Tue/... of afgekort)
            s = re.sub(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Ma|Di|Wo|Do|Vr|Za|Zo)\s+', '', s, flags=re.I)

            # Normaliseer maandnamen/afkortingen (voeg gerust varianten toe als je ze tegenkomt)
            MONTH_FIX = {
                # Engels
                r'\bSept\b': 'Sep',
                r'\bJun\.\b': 'Jun', r'\bJul\.\b': 'Jul', r'\bOct\.\b': 'Oct',
                # NL/DE/FR varianten die soms opduiken
                r'\bOkt\b': 'Oct', r'\bMei\b': 'May', r'\bMär\b': 'Mar', r'\bMärz\b': 'Mar',
                r'\bMai\b': 'May', r'\bJuni\b': 'Jun', r'\bJuli\b': 'Jul',
                r'\bJuin\b': 'Jun', r'\bJuil\b': 'Jul', r'\bAoût\b': 'Aug',
            }
            for pat, repl in MONTH_FIX.items():
                s = re.sub(pat, repl, s, flags=re.I)

            # Punten na maand afkortingen weg (bijv. "Sep." -> "Sep")
            # Verwijder punt na maand-afkortingen (Sep., Sept., Oct., etc.)
            s = re.sub(r'\b([A-Za-z]{3,4})\.\b', r'\1', s)
            # Verwijder eventuele komma direct na de maand (bijv. "6 Sep, 2025")
            s = re.sub(r',(?=\s|$)', '', s)
            
            # Jaar toevoegen als het ontbreekt
            now = datetime.now()
            if not re.search(r'\b\d{4}\b', s):
                s = f"{s} {now.year}"

            # Eerst strikt proberen met %d %b %Y
            try:
                dt = datetime.strptime(s, "%d %b %Y")
            except ValueError:
                # Fallback: dateutil is tolerant (pandas sleept dateutil meestal mee)
                from dateutil import parser
                # default zorgt dat ontbrekend jaar niet 1900 wordt
                default = datetime(now.year, 1, 1)
                dt = parser.parse(s, dayfirst=True, fuzzy=True, default=default)

            # Jaar-rollover (als we in dec zijn en we zien jan-datums, zet ze op volgend jaar)
            if now.month == 12 and dt.month == 1 and not re.search(r'\b\d{4}\b', raw_date):
                dt = dt.replace(year=now.year + 1)

            return dt.strftime("%Y-%m-%d")

        except Exception as e:
            logger.warning(f"Error parsing date '{raw_date}': {e}")
            return None


    def create_table(self):
        """Create eci_data table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS public.eci_data
        (
            match_id character varying NOT NULL,
            date character varying,
            home_team character varying,
            away_team character varying,
            home_win_pct double precision,
            draw_pct double precision,
            away_win_pct double precision,
            home_rating character varying,
            away_rating character varying,
            competition character varying,
            eci_score character varying,
            created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
            updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
            created_by character varying DEFAULT 'dekraan',
            CONSTRAINT eci_data_pkey PRIMARY KEY (match_id),
            CONSTRAINT eci_data_unique_match UNIQUE (date, home_team, away_team)
        )
        """
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()

    def clean_percentage(self, value):
        """Convert percentage string to float and round to 5 decimal places"""
        if pd.isna(value):
            return None
        try:
            # Convert string percentage to float
            if '%' in str(value):
                percentage = float(value.strip('%')) / 100
            else:
                percentage = float(value)
            # Round to 5 decimal places
            return round(percentage, 5)
        except (ValueError, TypeError):
            logger.warning(f"Error converting percentage value: {value}")
            return None

    def scrape_competition(self, driver, competition_name, competition_id):
        # Random pauze voor we een nieuwe competitie selecteren
        time.sleep(random.uniform(3, 7))
        
        select = Select(driver.find_element(By.CLASS_NAME, 'form-control'))
        select.select_by_value(competition_id)
        logger.info(f"Selected {competition_name} (ID: {competition_id})")

        # Korte pauze na selectie om de pagina te laten reageren
        time.sleep(random.uniform(2, 4))

        wait = WebDriverWait(driver, 20)
        try:
            # Check for "No Odds" message
            no_odds_elements = driver.find_elements(By.XPATH, "//h4[contains(text(), 'No Odds available yet...')]")
            if no_odds_elements:
                logger.info(f"No odds available for {competition_name}")
                return pd.DataFrame()

            # Wacht tot matches geladen zijn (bestaande conditionele wait)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'module-match-odds__item')))
            
            # Extra pauze om zeker te zijn dat alles geladen is
            time.sleep(random.uniform(2, 4))

            matches = driver.find_elements(By.CLASS_NAME, 'module-match-odds__item')
            if not matches:
                logger.info(f"No matches found for {competition_name}")
                return pd.DataFrame()

            data = []
            for match in matches:
                try:
                    # Date - Now with parsing
                    raw_date = match.find_element(By.CLASS_NAME, 'module-match-odds__item-date').text
                    date = self.parse_date(raw_date)
                    
                    if not date:
                        logger.info(f"Skipping match due to unparsable date: '{raw_date}' in {competition_name}")
                        continue

                    # Home team
                    home_team_full = match.find_element(
                        By.CSS_SELECTOR, 
                        '.module-match-odds__item-hometeam-info a'
                    ).text
                    home_rating_match = re.search(r'\(([-]?\d+)\)', home_team_full)
                    home_team = re.sub(r'\([-]?\d+\)', '', home_team_full).strip()
                    home_rating = home_rating_match.group(1) if home_rating_match else 'N/A'

                    # Bescherming tegen encoding-fouten zoals N?mme, BK H?cken, Brei?ablik.
                    # Als Selenium een teamnaam met '?' teruggeeft, slaan we deze rij niet op.
                    if "?" in home_team or "?" in away_team:
                        logger.warning(
                            f"Skipping suspicious encoding row in {competition_name}: "
                            f"{date} | {home_team} - {away_team}"
                        )
                        continue
                        
                    # Away team
                    away_team_full = match.find_element(
                        By.CSS_SELECTOR, 
                        '.module-match-odds__item-awayteam-info a'
                    ).text
                    away_rating_match = re.search(r'\(([-]?\d+)\)', away_team_full)
                    away_team = re.sub(r'\([-]?\d+\)', '', away_team_full).strip()
                    away_rating = away_rating_match.group(1) if away_rating_match else 'N/A'

                    # Score
                    score_elements = match.find_elements(By.CLASS_NAME, 'module-match-odds__item-score')
                    score = (score_elements[0].find_element(By.TAG_NAME, 'span').text 
                            if score_elements and score_elements[0].find_elements(By.TAG_NAME, 'span') 
                            else None)

                    # Percentages
                    home_win_pct = match.find_element(By.CLASS_NAME, 'module-match-odds__item-cup-home').text
                    draw_pct = match.find_element(By.CLASS_NAME, 'module-match-odds__item-draw').text
                    away_win_pct = match.find_element(By.CLASS_NAME, 'module-match-odds__item-cup-away').text

                    data.append([
                        date, home_team, home_rating, away_team, away_rating,
                        score, home_win_pct, draw_pct, away_win_pct
                    ])

                except Exception as e:
                    logger.exception(f"Error processing match in {competition_name}: {str(e)}")
                    continue

            return pd.DataFrame(data, columns=[
                'date', 'home_team', 'home_rating', 'away_team', 'away_rating',
                'score', 'home_win_pct', 'draw_pct', 'away_win_pct'
            ])

        except Exception as e:
            logger.exception(f"Error scraping {competition_name}: {str(e)}")
            return pd.DataFrame()

    def save_to_database(self, matches, competition):
        """Save scraped data to database (incl. eci_score)."""
        if matches.empty:
            return

        # map long -> short (keys van LEAGUES_CONFIG)
        short_name = None
        for key, value in LEAGUES_CONFIG.items():
            if value['name'] == competition:
                short_name = key
                break
        competition_name = short_name if short_name else competition

        with psycopg2.connect(**DB_CONFIG) as conn:
            for _, match in matches.iterrows():
                try:
                    with conn.cursor() as cursor:
                        match_id = f"{match['date']}_{match['home_team']}_{match['away_team']}"

                        odds_home = self.clean_percentage(match['home_win_pct'])
                        odds_draw = self.clean_percentage(match['draw_pct'])
                        odds_away = self.clean_percentage(match['away_win_pct'])
                        score_norm = normalize_score(match.get('score'))

                        cursor.execute("""
                            INSERT INTO public.eci_data (
                                match_id, date, home_team, away_team,
                                home_win_pct, draw_pct, away_win_pct,
                                home_rating, away_rating, competition,
                                eci_score, created_at, created_by
                            ) VALUES (
                                %s, %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s,
                                %s, %s, %s
                            )
                            ON CONFLICT ON CONSTRAINT eci_data_unique_match DO UPDATE SET
                                home_win_pct = EXCLUDED.home_win_pct,
                                draw_pct     = EXCLUDED.draw_pct,
                                away_win_pct = EXCLUDED.away_win_pct,
                                home_rating  = EXCLUDED.home_rating,
                                away_rating  = EXCLUDED.away_rating,
                                competition  = EXCLUDED.competition,
                                -- behoud bestaande score als de nieuwe None is
                                eci_score    = COALESCE(EXCLUDED.eci_score, eci_data.eci_score),
                                updated_at   = CURRENT_TIMESTAMP
                        """, (
                            match_id, match['date'], match['home_team'], match['away_team'],
                            odds_home, odds_draw, odds_away,
                            match['home_rating'], match['away_rating'], competition_name,
                            score_norm, self.current_timestamp, 'dekraan'
                        ))
                    conn.commit()
                except Exception as e:
                    logger.exception(f"Error saving match {match_id}: {str(e)}")
                    conn.rollback()
                    continue


    def run(self):
        logger.info("=== ECI SCRAPER RUN START ===")
        def init_driver():
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            
            # Verbergt de USB/GCM/DevTools meldingen in de terminal
            options.add_argument('--log-level=3')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.get(self.url)

            # Handle cookie consent
            try:
                cookie_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '.main-cookie-consent .btn'))
                )
                cookie_button.click()
                logger.info("Cookie consent accepted")
            except Exception:
                logger.info("No cookie consent button found")

            return driver

        driver = None
        try:
            driver = init_driver()

            # Wait for dropdown to be available
            wait = WebDriverWait(driver, 10)
            dropdown = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'form-control')))

            if not dropdown:
                raise Exception("Competition dropdown not found")

            # Process each competition
            for i, (key, league) in enumerate(LEAGUES_CONFIG.items()):
                logger.info(f"\nProcessing {league['name']}...")
                df = self.scrape_competition(driver, league['name'], league['id'])
                
                if not df.empty:
                    self.save_to_database(df, key)
                    logger.info(f"Saved {len(df)} matches for {league['name']} to database")
                else:
                    logger.info(f"No data to save for {league['name']}")
                
                time.sleep(random.uniform(5, 10))

                # Close and re-open the browser every 10 competitions
                if (i + 1) % 10 == 0:
                    driver.quit()
                    logger.info("Browser closed for a short break")
                    time.sleep(random.uniform(60, 120))
                    driver = init_driver()

            logger.info("\nAll data saved to database")

        except Exception as e:
            logger.exception(f"Fatal error: {str(e)}")
        finally:
            if driver is not None:
                driver.quit()
                logger.info("Browser closed")
                
            logger.info("=== ECI SCRAPER RUN END ===")

if __name__ == "__main__":
    scraper = ECIScraper()
    scraper.run()