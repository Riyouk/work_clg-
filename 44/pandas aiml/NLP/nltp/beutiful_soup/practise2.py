import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

# -----------------------------
# Setup
# -----------------------------
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")  # make sure installed: python -m spacy download en_core_web_sm

# -----------------------------
# 1. Get Article Text
# -----------------------------
def fetch_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return " ".join(paragraphs)
    else:
        print("Failed to fetch:", response.status_code)
        return ""

# -----------------------------
# 2. Preprocessing Functions
# -----------------------------
def clean_text(text):
    """Remove non-alphabetic characters and lowercase."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

def apply_stemming(tokens):
    return [stemmer.stem(t) for t in tokens]

def apply_lemmatization_nltk(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

def apply_lemmatization_spacy(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.text.lower() not in stop_words and token.is_alpha]

# -----------------------------
# 3. WordCloud Function
# -----------------------------
def make_wordcloud(tokens, title="WordCloud"):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.show()

# -----------------------------
# 4. Run the Pipeline
# -----------------------------
url = "https://en.wikipedia.org/wiki/Natural_language_processing"
article_text = fetch_article(url)

# Clean + Tokenize
cleaned_text = clean_text(article_text)
tokens = nltk.word_tokenize(cleaned_text)
tokens_no_stop = remove_stopwords(tokens)

# Versions
tokens_stemmed = apply_stemming(tokens_no_stop)
tokens_lemmatized_nltk = apply_lemmatization_nltk(tokens_no_stop)
tokens_lemmatized_spacy = apply_lemmatization_spacy(cleaned_text)

# -----------------------------
# 5. Visualize Word Clouds
# -----------------------------
make_wordcloud(tokens_no_stop, "After Stopword Removal")
make_wordcloud(tokens_stemmed, "After Stemming")
make_wordcloud(tokens_lemmatized_nltk, "After NLTK Lemmatization")
make_wordcloud(tokens_lemmatized_spacy, "After spaCy Lemmatization")
