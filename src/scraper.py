
import pandas as pd
import undetected_chromedriver as uc

def fetch_page(url):
    driver = uc.Chrome()
    driver.get(url)
    return driver

def get_table(driver, table_id):
    df = pd.read_html(driver.page_source, attrs = {'id': table_id})
    return df

if __name__ == "__main__":
    url = "https://fbref.com/en/comps/9/stats/Premier-League-Stats"

    driver = fetch_page(url)

    get_table(driver, "stats_squads_standard_for")
