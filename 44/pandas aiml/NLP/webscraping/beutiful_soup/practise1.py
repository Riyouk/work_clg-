# import requests
# from bs4 import BeautifulSoup

# # Step 1: Get the webpage
# url = "https://en.wikipedia.org/wiki/One_Piece"
# response = requests.get(url)

# # Step 2: Parse the HTML
# soup = BeautifulSoup(response.text, "html.parser")

# # Step 3: Extract all text
# text = soup.get_text()

# print(text)



import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/One_Piece"   # example site

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "html.parser")

print(soup.title.get_text())   # just to check

