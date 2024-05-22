import os
import csv
import time
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
from tqdm import tqdm

"""
How to operate:
- Check MS Edge version as driver and browser versions must be in sync to operate. [Current Version: 121.0.2277.128]

"""
def extract_info_from_string(input_string):
    # Define regex patterns to match the Journal name, year, and date
    journal_pattern = r'([A-Z][\w\s]+)\.'  # Match journal name ending with '.'
    year_pattern = r'(\d{4})'  # Match four digits for the year
    date_pattern = r'([A-Z][a-z]{2} \d{1,2})'  # Match month abbreviation followed by day

    # Search for matches in the input string
    journal_match = re.search(journal_pattern, input_string)
    year_match = re.search(year_pattern, input_string)
    date_match = re.search(date_pattern, input_string)

    # Extract matched groups
    journal = journal_match.group(1).strip() if journal_match else None
    year = year_match.group(1) if year_match else None
    date = date_match.group(1) if date_match else None

    # Convert the date format from "Oct 10" to "10/10/2018"
    if date:
        date_parts = date.split()
        if len(date_parts) == 2:
            day = date_parts[1]
            month_name = date_parts[0]
            month_abbr_to_num = {
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
            }
            month = month_abbr_to_num.get(month_name)
            if month:
                date = f"{day}/{month}/{year}"

    return journal, year, date

def extractor():
    startTime = time.time()
    page_count=int(input("What is the page count? -> "))
    content = []
    os.environ['PATH'] += r"C:/SeleniumDrivers"
    executable_path="C:/SeleniumDrivers/msedgedriver.exe"
    try:
        driver = webdriver.Edge(executable_path=executable_path)
        url="https://pubmed.ncbi.nlm.nih.gov/?term=Host+immune+response+to+Dengue+infection"
        driver.get(url)
    except Exception as e:
        print(f"Driver initialization unsuccessful.{e}")
    os.makedirs("output", exist_ok=True)
    #button = driver.find_element_by_class_name('next-page-btn')
    
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

           #content.append(html_content) 
            print(type(soup))
            count=count+1
            print(f"\nNew count: {count}. --- Current Sub Count: {sub_count}.")
            if count>sub_count and count <= page_count:
                button = driver.find_element_by_class_name('next-page-btn')
                button.click()
                sub_count=sub_count+1
                print(f"\nNew Subcount: {sub_count}")
            else:
                driver.quit()
                break
    
    print(f"\n\n\n-----HTML EXTRACTION COMPLETE-----")
    print(f"\n-----HTML PARSING STARTED-----\n\n\n")

    html_files_directory = "D:\Visual Studio\CSE499\Webscraper\output_2"
    html_output_dir = "D:\Visual Studio\CSE499\Webscraper\output_2_csv"

    os.makedirs(html_output_dir, exist_ok=True)

    data = []
    for filename in os.listdir(html_files_directory):
        if filename.endswith('.html'):
            file_path = os.path.join(html_files_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content=file.read()

            soup = BeautifulSoup(content, 'lxml')

            article_elements = soup.find_all('article', class_='full-docsum')

            for article in article_elements:
                # Extract specific information from each article
                title_element = article.find('a', class_='docsum-title')
                title = title_element.get_text(strip=True)

                # Extract the link (URL) using the get method
                link = title_element.attrs.get('href', '')
                authors = article.find('span', class_='docsum-authors').get_text(strip=True)
                journal_citation = article.find('span', class_='docsum-journal-citation full-journal-citation').get_text(strip=True)
                journal, year, date = extract_info_from_string(journal_citation)
                pmid = article.find('span', class_='docsum-pmid').get_text(strip=True)

                # Link Processing
                url_prefix="https://pubmed.ncbi.nlm.nih.gov"
                url=url_prefix+link
                # Print or store the extracted information
                data.append({"PMID": pmid, "Title": title,"URL": url, "Authors": authors, "Date": date, "Year": year, "Journal":journal})
                print(title)

        # Create a DataFrame from the data
    df = pd.DataFrame(data)
    csv_filename= 'all_data_2.csv'
    # Save to CSV in the output directory
    df.to_csv(csv_filename, index=False)
    print(f"\n\n-----PARSED ELEMENTS STORED-----")
    file_path = os.path.join(html_output_dir, csv_filename)

    content = {}
    column_name = 'URL'
    print(f"Column Chosen:{column_name}.\n Here are the top 5 {column_name}s: \n{df[column_name].head()}")
    all_links = df[column_name]
    print(f"All Links Have been Transferred Successfully. \nLength of the column list: {len(all_links)}")
    total_links = len(all_links)
    index = 0

    for index in tqdm(range(total_links), desc=f"Scraping Progress:"):
        url = all_links[index]
        driver.get(url)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content,'lxml')
        title_element = soup.find('h1', class_='heading-title')
        abstract_element = soup.find('div', class_='abstract-content')
        #year_element = soup.find('span', class_='docsum-journal-citation full-journal-citation')
        if title_element:
            title_text = title_element.get_text(strip=True)
            if abstract_element and title_element:
                abstract_text = abstract_element.get_text(strip=True)
            else:
                abstract_text = "no_abstract_available"
        else:
            title_text = "no_title_available"
            abstract_text = "NaN"
        #print(abstract_text)
        driver.implicitly_wait(10)
        driver.quit 

        pmid = os.path.basename(url.rstrip('/'))

        content[pmid] = abstract_text

    df['Abstract'] = df['PMID'].map(content)

    output_file = os.path.join(html_output_dir, 'updated_file.csv')
    df.to_csv(output_file, index=False)

    endTime = time.time()
    elapsedTime = endTime - startTime
    print("Elapsed Time = %s" % elapsedTime)







extractor()
