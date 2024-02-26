import os
import csv
import time
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC

"""
How to operate:
- Check MS Edge version as driver and browser versions must be in sync to operate. [Current Version: 121.0.2277.128]

"""
def main():
    page_count=int(input("What is the page count?:"))

    os.environ['PATH'] += r"C:/SeleniumDrivers"
    executable_path="C:/SeleniumDrivers/msedgedriver.exe"
    try:
        driver = webdriver.Edge(executable_path=executable_path)
        url="https://pubmed.ncbi.nlm.nih.gov/?term=Host+immune+response+to+Dengue+infection"
        driver.get(url)
    except Exception as e:
        print(f"Driver initialization unsuccessful.{e}")
    os.makedirs("output", exist_ok=True)
    button = driver.find_element_by_class_name('next-page-btn')
    
    count=1
    sub_count = count
    while(True):
        if count <= page_count and count <= sub_count:
            driver.implicitly_wait(5)
            html_content = driver.page_source
            soup = BeautifulSoup(html_content,'lxml')
            time_count=int(time.time())
            output_file_path = f'output_2\parsed_html_{time_count}.html'
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(soup.prettify())
            count=count+1
            if count>sub_count:
                button = driver.find_element_by_class_name('next-page-btn')
                button.click()
                sub_count=count+1
        else:
            driver.quit()
            break
    


main()
