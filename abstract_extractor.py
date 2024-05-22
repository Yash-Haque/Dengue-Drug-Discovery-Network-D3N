import os
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
driver = webdriver.Edge(executable_path=executable_path)

# def extract(index: int, batch_range: int, batch_idx: int, all_links: list, content: list) -> list:
#     try:
#         # Need a function method for this entire procedure.
#         for index in tqdm(range(batch_range), desc=f"Scraping Progress for {batch_idx}"):
#             print(index)
#             url = all_links[index]
#                 #print(f"Scraping url: {url}")
#             driver.get(url)
#             html_content = driver.page_source
#             soup = BeautifulSoup(html_content,'lxml')
#             title_element = soup.find('h1', class_='heading-title')
#             abstract_element = soup.find('div', class_='abstract-content')
#             if title_element:
#                 title_text = title_element.get_text(strip=True)
#                 if abstract_element and title_element:
#                     abstract_text = abstract_element.get_text(strip=True)
#                 else:
#                     abstract_text = "no_abstract_available"
#             else:
#                 title_text = "no_title_available"
#                 abstract_text = "NaN"
#             #print(abstract_text)
#             driver.implicitly_wait(10)
#             driver.quit 
                
#             pmid = os.path.basename(url.rstrip('/'))

#             content.append({"Title": title_text,"PMID": pmid, "Abstract": abstract_text})
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     return content

def main():

    folder_path = "D:\Visual Studio\CSE499\Webscraper\output_2_csv"
    #file_name = "all_data_2.csv"
    file_path = "D:\\Visual_Studio\\CSE499\\Webscraper\\all_data_2.csv"

    #count=0
    content = []
    column_name = 'URL'
    #new_column_name = 'Abstract'
    try:
        startTime = time.time()
        df = pd.read_csv(file_path)
        print(f"Column Chosen:{column_name}.\n Here are the top 5 {column_name}s: \n{df[column_name].head()}")
        all_links = df[column_name]
        #all_titles = df['Title']
        print(f"All Links Have been Transferred Successfully. \nLength of the column list: {len(all_links)}")
        # batch_len = math.ceil(len(all_links)/10)
        total_links = len(all_links)
        dataset_array = np.arange
        # print(f"Batch Length: {batch_len}")

        batch_start: int = 1 # Default 0
        index: int = batch_start * 1000
        # Driver Initialization
        
        os.makedirs("output_2_csv", exist_ok=True)

        # for batch_idx in range(10):
        #     batch_range = index + 1000
        #     content = extract(index,batch_range, batch_idx, all_links, content)
        #     if not batch_idx==0:
        #         batch_range = index + 1000 + 1
        #         content = extract(index,batch_range, batch_idx, all_links, content)
        #         #print(f"Abstract of {url} added to content list.")
                
        for index in tqdm(range(total_links), desc=f"Scraping Progress:"):
            url = all_links[index]
            #print(f"Scraping url: {url}")
            driver.get(url)
            html_content = driver.page_source
            soup = BeautifulSoup(html_content,'lxml')
            title_element = soup.find('h1', class_='heading-title')
            abstract_element = soup.find('div', class_='abstract-content')
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

            content.append({"PMID":pmid, "Abstract": abstract_text})
             
            
        dataframe = pd.DataFrame(content)
        print(f"Here are the first five:\n{df.head()}")
        print(f"Printing first 5 contents: {content[3]}")
        #csv_filename = os.path.join(folder_path, 'with_abstracts_2.csv')
        dataframe.to_csv("only_abstracts_2", index=False)

        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed Time = %s" % elapsedTime)


    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
