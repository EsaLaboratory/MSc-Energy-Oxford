from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time


def scrape_nordpool(date):
# Set up Firefox driver using GeckoDriverManager
    driver = webdriver.Firefox()

    # URL of the web page with the table
    url = 'https://data.nordpoolgroup.com/auction/gb-half-hour/prices?deliveryDate='+str(date)+'&currency=GBP&aggregation=Hourly&deliveryAreas=UK'  # Replace with the actual URL

    # Open the web page
    driver.get(url)

    try:
        # Wait for the accept cookies button to be clickable
        accept_cookies_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[2]/div/mat-dialog-container/div/div/gdpr-dialog/div/div[2]/div/button[2]')))
        
        # Click the accept cookies button
        accept_cookies_button.click()
        
        # Wait a bit for the banner to disappear (optional)
        time.sleep(2)  # Adjust as needed

    except Exception as e:
        print(f"Exception occurred while accepting cookies: {str(e)}")

    driver.execute_script("window.scrollTo(0, 550);")  # Adjust the scroll amount as needed

    try:
        # Wait for the table to be visible
        table = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, 'dx-datagrid-table-fixed')))
        
        # Find all rows in the table body
        rows = table.find_elements(By.XPATH, "//tbody/tr[@role='row']")
        
        prices = []
        for row in rows[2:50]:
            # Extract time interval and number from each row
            number = row.find_element(By.XPATH, ".//td[@aria-colindex='2']").text.strip()
            
            prices.append(float(str(number).replace(',','.')))
            
    finally:
        driver.quit()

    return prices

prices = scrape_nordpool('2023-11-15')
print(prices)