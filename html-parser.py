import os
from bs4 import BeautifulSoup
import pandas as pd
import re

html_files_directory = 'D:\Visual_Studio\CSE499\Webscraper\output_2'
output_directory = 'D:\Visual_Studio\CSE499\Webscraper\output_2_csv' 
os.makedirs(output_directory, exist_ok=True)

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
            journal_citation = article.find('span', class_='docsum-journal-citation').get_text(strip=True)
            print(type(journal_citation))
            journal,year,date = extract_info_from_string(journal_citation)
            pmid = article.find('span', class_='docsum-pmid').get_text(strip=True)

            # Link Processing
            url_prefix="https://pubmed.ncbi.nlm.nih.gov"
            url=url_prefix+link
            # Print or store the extracted information
            data.append({"PMID": pmid, "Title": title,"URL": url, "Authors": authors,"Date":date, "Year": year,"Journal":journal })
            print(title)

    # Create a DataFrame from the data
df = pd.DataFrame(data)
csv_filename= 'all_data_2.csv'
# Save to CSV in the output directory
df.to_csv(csv_filename, index=False)

print(f"Data from all HTML files saved to {csv_filename}")