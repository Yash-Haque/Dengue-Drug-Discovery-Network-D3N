import os
import traceback
import pandas as pd
import time
import numpy as np
import math
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC


os.environ['PATH'] += r"C:/SeleniumDrivers"
executable_path="C:\\SeleniumDrivers\\msedgedriver.exe"

service= Service(executable_path=executable_path)


def main():
    output_path:str = "D:\\Visual_Studio\\CSE499\\Webscraper\\output\\parsed_data\\staging"
    input_path:str = "D:\\Visual_Studio\\CSE499\\Webscraper\\output\\parsed_data\\staging\\parse-1709891135.csv"
    
    try:
        
        df = pd.read_csv(input_path)

        titles = []
        abstracts = []
        authors_list = []
        journal_names = []
        publications = []
        pmids = []
        dois = []
        keywords_lists = []
        similar_pmids_lists = []
        similar_titles_lists = []

        startTime = time.time()

        

        for index, row in tqdm(df.iterrows(), total=len(df)): # Iteration starts here
            # Driver Initialization
            url = row['url']
            driver = webdriver.Edge(service=service)
            driver.get(url)

            # Web crawling
            html_content = driver.page_source

            # Html Parsing
            soup = BeautifulSoup(html_content,'lxml')

            # Extracting Titles
            heading_title = soup.find('h1', class_='heading-title')
            title_text = heading_title.get_text(strip=True)
            #print("Title:", title_text)
            titles.append(title_text)

            # Extracting Abstracts
            abstract_content = soup.find('div', class_= "abstract-content selected")
            abstract = abstract_content.get_text(strip=True)
            #print(f"Abstract: {abstract}")
            abstracts.append(abstract)

            # Extracting Authors
            authors = soup.find('div', class_='authors')
            author_spans = authors.find_all('span', class_='authors-list-item')

            for span in author_spans:
                author_name = span.find('a', class_='full-name').text
                authors_list.append(author_name)
            #print("Authors:", authors_list)

            # Extracting Journals
            journal_trigger = soup.find('button', id='full-view-journal-trigger')
            journal_name = journal_trigger.get_text(strip=True)
            #print(f"Journal Name: {journal_name}")
            journal_names.append(journal_name)

            # Extracting Journal Name and Time
            cit = soup.find('span', class_='cit')
            publication = cit.get_text(strip=True)
            #print("Journal and Time:", publication)
            publications.append(publication)

            # Extracting pmids
            id = soup.find('strong', class_='current-id')
            pmid = int(id.get_text(strip=True))
            #print(f"PMID: {pmid}")
            pmids.append(pmid)

            # Extracting DOIs
            id_link = soup.find('a', class_='id-link')
            doi = id_link.get_text(strip=True)
            #print(f"DOI: {doi}, DOI Link: {doi_link}")
            dois.append(doi)

            # Extracting Keywords
            sub_title = soup.find('strong', class_='sub-title')
            p_element = sub_title.parent
            keywords_text = p_element.get_text(strip=True)
            keywords = keywords_text.split(';')
            prefix_len = len("Keywords:")
            keywords[0] = keywords[0][prefix_len:].strip()
            #print(f"Keywords: {keywords}") 

            # Extracting Similar Articles
            similar_articles = soup.find_all('li', class_='full-docsum')
            similar_titles = []
            similar_pmids = []
            # Extracting pmids and titles
            for article in similar_articles:
                link = article.find('a', class_='docsum-title')
                title = link.get_text(strip=True)
                similar_titles.append(title)
                
                pmid_element = article.find('span', class_='docsum-pmid')
                docsum_pmid = pmid_element.get_text(strip=True)
                similar_pmids.append(docsum_pmid)
            
            similar_titles_lists.append(similar_titles)
            similar_pmids_lists.append(similar_pmids)

        driver.quit()

        data = {"pmid":}

    except Exception as e:
        print("\nAn error occurred:\n", e)
        traceback.print_exc()


if __name__ == "__main__":
   main()

