import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Specify the URL
url = 'https://www.crunchbase.com/lists/angel-investors-pre-seed-list-no-direct/05f45676-9ad8-49d3-810c-120290bb88c6/principal.investors'  # Replace with your Crunchbase list URL

# Step 2: Send an HTTP request to the URL
response = requests.get(url)

# Step 3: Parse the webpage content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Step 4: Find the relevant data
data = []
for item in soup.find_all('div', class_='card-content__content-wrapper'):
    name = item.find('a', class_='name').text.strip() if item.find('a', class_='name') else 'N/A'
    detail = item.find('span', class_='component--field-formatter').text.strip() if item.find('span', class_='component--field-formatter') else 'N/A'
    data.append([name, detail])

# Step 5: Create a DataFrame using pandas
df = pd.DataFrame(data, columns=['Name', 'Detail'])

# Step 6: Save the DataFrame to an Excel file
df.to_excel('crunchbase_list.xlsx', index=False)

print("Data scraped and saved to crunchbase_list.xlsx")
