from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
import requests
import time
import random
import csv
from io import StringIO
import pandas as pd
import os
import logging
import re
from datetime import datetime, timedelta

# project directory (use cwd by default)
PRJ_DIR = os.path.abspath('')
# PRJ_DIR = os.path.join(PRJ_DIR, 'code')
# basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# urls of EPL matches from the last 5 seasons
IDlist = {
    '2018-2019' : range(1284741, 1285121),
    '2019-2020' : range(1375927, 1376307),
    '2020-2021' : range(1485184, 1485564),
    '2021-2022' : range(1640674, 1641054), 
    '2022-2023' : range(1641049, 1641054),
    '2023-2024' : range(1729190, 1729570),
    '2024-2025' : range(1821049, 1821429),
    '2025-2026' : range(1903207, 1903268),
    # 'manually': [1729448, 1729492]
}

# path for pregame data and event data
pregame_dir = os.path.join(PRJ_DIR, 'data', 'pregame_data')
event_dir = os.path.join(PRJ_DIR, 'data', 'event_data')
if not os.path.exists(pregame_dir):
    os.makedirs(pregame_dir, exist_ok=True)
if not os.path.exists(event_dir):
    os.makedirs(event_dir, exist_ok=True)
csv_file = os.path.join(pregame_dir, 'pregame_data.csv')

# attributes for pregame data
keylist = ['match_id', 'date', 'home_team', 'away_team', 'home_team_id', 'away_team_id', 'home_team_elo', 'away_team_elo']

# Create header if file doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keylist)
        writer.writeheader()
    logging.info(f"Created new CSV file: {csv_file}")

for season, urls in IDlist.items():
    logging.info(f"Processing season: {season}")
    
    for matchID in urls:
        matchID = str(matchID)
        url = 'https://1xbet.whoscored.com/matches/' + matchID + '/live'
        # Check if event CSV already exists for this match
        event_file_path = os.path.join(event_dir, matchID + '.csv')
        event_exists = os.path.exists(event_file_path)
        time.sleep(random.uniform(2, 5))  # polite delay between requests
        driver = None
        try:
            # open web driver on the url
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            # chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            # chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            driver = webdriver.Chrome(options=chrome_options)
            # driver = webdriver.Chrome()
            driver.get(url)

            # Find the script containing matchCentreData (it's not always in the same position)
            wait = WebDriverWait(driver, 30)
            
            # Search through all script tags for matchCentreData
            scripts = driver.find_elements(By.TAG_NAME, 'script')
            data_text = None
            
            for script in scripts:
                script_content = script.get_attribute('innerHTML')
                if script_content and 'matchCentreData' in script_content:
                    data_text = script_content
                    logging.info(f"Match {matchID}: Found matchCentreData in script tag")
                    break
            
            if not data_text:
                logging.warning(f"Match {matchID}: matchCentreData not found in any script tag; skipping")
                continue

            # Extract the matchCentreData JSON
            start_index = data_text.find('matchCentreData')
            if start_index == -1:
                logging.warning(f"Match {matchID}: matchCentreData keyword not found; skipping")
                continue

            # Extract JSON substring - find the opening { and matching }
            data_sub = data_text[start_index + len('matchCentreData:'):]
            
            # Find the JSON object boundaries
            brace_start = data_sub.find('{')
            if brace_start == -1:
                logging.warning(f"Match {matchID}: Could not find JSON start; skipping")
                continue
            
            # Count braces to find the matching closing brace
            brace_count = 0
            json_end = -1
            for i, char in enumerate(data_sub[brace_start:], start=brace_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end == -1:
                logging.warning(f"Match {matchID}: Could not find JSON end; skipping")
                continue
            
            json_str = data_sub[brace_start:json_end]

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(f"Match {matchID}: JSON decode error: {e}; skipping")
                continue

            # extract event data (may be missing). Only save if not already present
            event_data = data.get('events', [])
            if event_data and not event_exists:
                df = pd.DataFrame(event_data)
                df.to_csv(event_file_path, index=False)
                logging.info(f"Match {matchID}: event data saved to {event_file_path}")
            elif event_exists:
                logging.info(f"Match {matchID}: event CSV already exists; skipping event save")
            else:
                logging.info(f"Match {matchID}: no event data found")

            # get the match date to crawl ELO rating data
            start_time = data.get('startTime')
            if not start_time or not isinstance(start_time, str):
                logging.warning(f"Match {matchID}: missing or invalid startTime; skipping")
                continue
            # startTime expected like 'YYYY-MM-DDTHH:MM:SSZ' -> take date portion
            date = start_time.split('T')[0]
            try:
                ddate = datetime.strptime(date, '%Y-%m-%d')
            except Exception as e:
                logging.warning(f"Match {matchID}: date parse error '{date}': {e}; skipping")
                continue

            previous_day = ddate - timedelta(days=1)
            predate = previous_day.strftime('%Y-%m-%d')

            # get data from clubelo api
            try:
                r = requests.get('http://api.clubelo.com/' + predate, timeout=10)
                if r.status_code != 200:
                    logging.warning(f"ClubElo API returned {r.status_code} for {predate}; elo lookup skipped")
                    elo = pd.DataFrame()
                else:
                    elo_data = StringIO(r.text)
                    elo = pd.read_csv(elo_data, sep=",")
            except Exception as e:
                logging.warning(f"Match {matchID}: ClubElo request failed: {e}")
                elo = pd.DataFrame()

            # Extract team names from JSON data (no need to scrape HTML)
            home_team_data = data.get('home', {})
            away_team_data = data.get('away', {})
            
            home_team_name = home_team_data.get('name', '').strip()
            away_team_name = away_team_data.get('name', '').strip()
            
            if not home_team_name or not away_team_name:
                logging.warning(f"Match {matchID}: unable to extract team names from JSON; skipping")
                continue

            # normalize names mapping
            name_map = {
                'Man Utd': 'Man United',
                'WBA': 'West Brom',
                'Sheff Utd': 'Sheffield United',
                'Nottingham Forest': 'Forest'
            }
            home_team_name = name_map.get(home_team_name, home_team_name)
            away_team_name = name_map.get(away_team_name, away_team_name)

            # safe elo lookup helper
            def get_elo(club_name):
                try:
                    vals = elo.loc[elo['Club'] == club_name, 'Elo']
                    if vals.empty:
                        return None
                    return vals.values[0]
                except Exception:
                    return None

            home_team_elo = get_elo(home_team_name)
            away_team_elo = get_elo(away_team_name)

            # store pregame data
            instance_data = {
                'match_id': matchID,
                'date': date,
                'home_team': home_team_name,
                'away_team': away_team_name,
                'home_team_id': data.get('home', {}).get('teamId'),
                'away_team_id': data.get('away', {}).get('teamId'),
                'home_team_elo': home_team_elo,
                'away_team_elo': away_team_elo
            }

            # Write immediately to CSV (append mode)
            with open(csv_file, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=keylist)
                writer.writerow(instance_data)
            
            logging.info(f"Match {matchID}: Successfully scraped and saved to {csv_file}")

        except TimeoutException as e:
            logging.warning(f"Match {matchID}: timeout waiting for element: {e}; skipping")
            continue
        except Exception as e:
            logging.exception(f"Match {matchID}: unexpected error: {e}")
            continue
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

logging.info("Scraping complete!")