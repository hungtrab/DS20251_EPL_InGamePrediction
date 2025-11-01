from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time

# Test with a known match ID
matchID = '1640674'
url = f'https://1xbet.whoscored.com/matches/{matchID}/live'

chrome_options = Options()
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

driver = webdriver.Chrome(options=chrome_options)

try:
    print(f"[INFO] Opening URL: {url}")
    driver.get(url)
    
    # Wait for page to load
    time.sleep(5)
    
    print("\n[INFO] Searching for all <script> tags...")
    scripts = driver.find_elements(By.TAG_NAME, 'script')
    
    print(f"[INFO] Found {len(scripts)} script tags\n")
    
    for i, script in enumerate(scripts):
        script_content = script.get_attribute('innerHTML')
        if script_content and len(script_content) > 0:
            # Check if it contains match data keywords
            if any(keyword in script_content for keyword in ['matchCentreData', 'matchData', 'eventData', 'incidents', 'teamData']):
                print(f"=" * 80)
                print(f"[FOUND] Script #{i} contains potential match data")
                print(f"Location: {script.get_attribute('outerHTML')[:200]}")
                print(f"Length: {len(script_content)} characters")
                print(f"\nFirst 500 characters:")
                print(script_content[:500])
                print(f"\n[Search for]: 'matchCentreData' = {'matchCentreData' in script_content}")
                print(f"[Search for]: 'matchData' = {'matchData' in script_content}")
                print(f"[Search for]: 'events' = {'events' in script_content}")
                print(f"=" * 80 + "\n")
    
    # Also check for data in specific div elements
    print("\n[INFO] Checking for JSON in div elements...")
    try:
        divs = driver.find_elements(By.XPATH, "//div[@id or @data-json or contains(@class, 'data')]")
        for div in divs[:10]:  # Check first 10
            content = div.get_attribute('innerHTML')
            if content and 'match' in content.lower():
                print(f"DIV with potential data: {div.get_attribute('outerHTML')[:200]}")
    except Exception as e:
        print(f"No special divs found: {e}")
    
    # Save full page source for manual inspection
    with open('debug_page_source.html', 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print("\n[INFO] Full page source saved to 'debug_page_source.html'")
    
    # Check page title to confirm we're on the right page
    print(f"\n[INFO] Page title: {driver.title}")
    
finally:
    driver.quit()
    print("\n[INFO] Done! Check the output above for the location of match data.")