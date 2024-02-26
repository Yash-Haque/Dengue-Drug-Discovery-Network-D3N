import csv

def load_text_file(text_file_path):
    """
    Load data from a text file into a dictionary.

    Args:
        text_file_path (str): Path to the text file.

    Returns:
        dict: A dictionary containing data from the text file.
    """
    text_data = {}
    with open(text_file_path, 'r', encoding='utf-8') as text_file:
        next(text_file)  # Skip the header row
        for line in text_file:
            parts = line.strip().split(',')  # Assuming comma-separated values
            pmid = int(parts[0])
            abstract = ','.join(parts[1:])  # Reconstruct abstract from remaining parts
            text_data[pmid] = abstract
    return text_data

def load_csv_file(csv_file_path):
    """
    Load data from a CSV file into a dictionary.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary containing data from the CSV file.
    """
    csv_data = {}
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            pmid = int(row['PMID'])
            csv_data[pmid] = {
                'Title': row['Title'],
                'URL': row['URL'],
                'Authors': row['Authors'],
                'Date': row['Date'],
                'Year': int(row['Year']),
                'Journal': row['Journal'],
                'Abstract': ''  # Initialize Abstract to an empty string
            }
    return csv_data

def merge_abstracts(text_data, csv_data):
    """
    Merge abstracts from the text file into the CSV data using PMID as a reference.

    Args:
        text_data (dict): Dictionary containing data from the text file.
        csv_data (dict): Dictionary containing data from the CSV file.

    Returns:
        dict: Merged dictionary with abstracts added to the CSV data.
    """
    for pmid, abstract in text_data.items():
        if pmid in csv_data:
            csv_data[pmid]['Abstract'] = abstract
    return csv_data


import csv

def save_merged_data_to_csv(merged_data, output_file):
    """
    Save merged data to a CSV file.

    Args:
        merged_data (dict): Merged data dictionary.
        output_file (str): Path to the output CSV file.
    """
    fieldnames = ['PMID', 'Title', 'URL', 'Authors', 'Date', 'Year', 'Journal', 'Abstract']
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pmid, data in merged_data.items():
            writer.writerow({'PMID': pmid,
                             'Title': data['Title'],
                             'URL': data['URL'],
                             'Authors': data['Authors'],
                             'Date': data['Date'],
                             'Year': data['Year'],
                             'Journal': data['Journal'],
                             'Abstract': data['Abstract']})




# Paths to the text and CSV files
text_file_path = 'D:\Visual_Studio\CSE499\Webscraper\only_abstracts_2.txt'
csv_file_path = 'D:\\Visual_Studio\\CSE499\\Webscraper\\all_data_2.csv'

# Load data from the text file
text_data = load_text_file(text_file_path)

# Load data from the CSV file
csv_data = load_csv_file(csv_file_path)

# Merge abstracts from the text file into the CSV data
merged_data = merge_abstracts(text_data, csv_data)

# Path to the output CSV file
output_file = 'D:\Visual_Studio\CSE499\Webscraper\merged_data.csv'

# Save merged data to CSV
save_merged_data_to_csv(merged_data, output_file)

# Print or further process the merged data
for pmid, data in merged_data.items():
    print(f"PMID: {pmid}, Abstract: {data['Abstract']}")  # Print or process as needed

