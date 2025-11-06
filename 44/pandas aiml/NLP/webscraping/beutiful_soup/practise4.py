from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://www.animenewsnetwork.com/"

def get_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text,"html.parser")
    return soup

soup = get_page(url)

headlines = []

for h in soup.find_all('h2'):
    title = h.text.strip()
    if title:
        headlines.append(title)
    
df = pd.DataFrame(headlines, columns=['Headlines'])
print(df)