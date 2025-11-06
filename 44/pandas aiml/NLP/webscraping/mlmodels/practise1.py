import requests
from newspaper import Article
import spacy

def smart_news_scraper(url):
    # Step 1: Extract article content
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    # Step 2: NLP model for named entities
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article.text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Step 3: Combine output
    data = {
        "Title": article.title,
        "Authors": article.authors,
        "Date": str(article.publish_date),
        "Summary": article.summary,
        "Named Entities": entities
    }
    return data

# Try it
url = "https://www.bbc.com/news/world-67285550"
result = smart_news_scraper(url)

for key, value in result.items():
    print(f"\n🔹 {key}:")
    print(value)
