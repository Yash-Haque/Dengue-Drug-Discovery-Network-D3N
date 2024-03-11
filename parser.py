import os
import time
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException


class Parser:
    def __init__(self, data):
        self.data = data
        os.environ['PATH'] += r"C:/SeleniumDrivers"
        self.executable_path = "C:/SeleniumDrivers/msedgedriver.exe"
        self.service = Service(executable_path=self.executable_path)
        self.output_folder = "output/parsed_htmls/staging"
        os.makedirs(self.output_folder, exist_ok=True)
        self.parsed_output_folder = "output/parsed_data/staging"
        os.makedirs(self.parsed_output_folder, exist_ok=True)
        self.parsed_data = []
        self.__run()

    def __run(self):
        try:
            url_prefix = "https://pubmed.ncbi.nlm.nih.gov/"
            url_path = "?term="
            for url, page_count in self.data.items():
                formatted_url = url_prefix + url_path + url.replace(" ", "+")
                print(f"Processing URL: {formatted_url}")
                print(f"Page Count: {page_count}")
                driver = webdriver.Edge(service=self.service)
                driver.get(formatted_url)
                for _ in tqdm(range(page_count)):
                    self.parse_page(driver)
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.CLASS_NAME, 'next-page-btn'))).click()
                    except TimeoutException:
                        print("Next page button not found or clickable within 10 seconds.")
                        break
                driver.quit()
            df = pd.DataFrame(self.parsed_data)
            time_count = int(time.time())
            csv_filename = f'parse-{time_count}.csv'
            filepath = os.path.join(self.parsed_output_folder, csv_filename)
            df.to_csv(filepath, index=False)
        except Exception as e:
            print(f"An error occurred: {e}")

    def parse_page(self, driver):
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'lxml')
        article_elements = soup.find_all('article', class_='full-docsum')
        for article in article_elements:
            title_element = article.find('a', class_='docsum-title')
            link = title_element.attrs.get('href', '')
            pmid = article.find('span', class_='docsum-pmid').get_text(strip=True)
            url = f"https://pubmed.ncbi.nlm.nih.gov{link}"
            self.parsed_data.append({"pmid": pmid, "url": url})


if __name__ == "__main__":
    input_dict = {}
    urls_count = int(input("How many URLs do you want to process?: "))
    for i in range(urls_count):
        url = input(f"Enter URL {i+1}: ")
        page_count = int(input(f"Enter the page count for URL {i+1}: "))
        input_dict[url] = page_count
    parser = Parser(input_dict)
