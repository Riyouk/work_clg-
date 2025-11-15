import requests
from bs4 import BeautifulSoup

url = "https://www.engadget.com/reviews/wearables/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

titles = []
summaries = []

for article in soup.find_all("span"):
    title = article.get_text(strip=True)
    if title:
        titles.append(title)
# print(titles)
import pandas as pd 
df = pd.DataFrame({"Title": titles})
# print(df.head(10))

# # ... existing code ...
# min_len = min(len(titles), len(summaries))
# df = pd.DataFrame({
#     "Title": [t.strip() for t in titles[:min_len]],
#     "Summary": [s.strip() for s in summaries[:min_len]],
# })

# print(df.head(10))


#text preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(tokens)
df["cleaned_title"] = df["Title"].apply(preprocess_text)
print(df["cleaned_title"].head(10))

# def preprocess(text):
#     text = text.lower()
#     tokens = word_tokenize(text)
#     cleaned_words = []
#     for token in tokens:
#         if token in stop_words or token in string.punctuation or token.isnumeric():
#             continue
#         cleaned_words.append(token)
#     return " ".join(cleaned_words)

# df["cleaned_title"] = df["Title"].apply(preprocess)
# print(df["cleaned_title"].head(10))


# labeling and furthur nlp tasks

def auto_label(text):
    text = text.lower()
    postive_keywords = ["good", "great", "excellent", "positive", "wonderful", "amazing", "fantastic", "superb", "outstanding", "terrific","love","like"]
    negative_keywords = ["bad", "poor", "negative","unfortunately","disappointing","hate","dislike","unhappy","worst","terrible","disappointing","unpleasant","dissatisfied","slow"]
    
    score = 0
    for word in text.split():
        if word in postive_keywords:
            score += 1
        elif word in negative_keywords:
            score -= 1
            
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"    

df["sentiment"] = df["cleaned_title"].apply(auto_label)
print(df["sentiment"].head(10))