import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import re
from wordcloud import WordCloud
from collections import Counter

# Downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# -----------------------------
# Preprocessing Functions
# -----------------------------
def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, remove stopwords."""
    text = re.sub('[^a-zA-Z]', ' ', text)  
    text = text.lower()
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return words

def stem_and_lemmatize(words):
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_words = [ps.stem(w) for w in words]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return stemmed_words, lemmatized_words

def generate_ngrams(words, n=2):
    return list(ngrams(words, n))

def pos_tagging(words):
    return nltk.pos_tag(words)

def word_cloud(words, title="Word Cloud"):
    wc_pos = WordCloud(width=800, height=400, background_color='white',
                       colormap='Greens', max_words=200).generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def top_frequencies(words, n=10):
    counter = Counter(words)
    return counter.most_common(n)

def read_file(file_path):
    with open(file_path,'r') as f:
        txt = f.read()
    return txt


def main():
    opt = input("Enter text to preprocess (or type 'file' to read from file): ")
    if opt.strip().lower() == 'file':

        file_path = "C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/manual/text.txt"
        txt = read_file(file_path)   # pass string, not file object

        words = preprocess_text(txt)
        print("\nPreprocessed Words:", words)

        stemmed, lemmatized = stem_and_lemmatize(words)
        print("\nStemmed Words:", stemmed[:20])   # preview first 20
        print("\nLemmatized Words:", lemmatized[:20])

        bigrams = generate_ngrams(words, 2)
        trigrams = generate_ngrams(words, 3)
        print("\nBigrams:", bigrams[:10])   # preview 10
        print("\nTrigrams:", trigrams[:10])

        pos_tags = pos_tagging(words)
        print("\nPOS Tags:", pos_tags[:20])

        print("\nTop Word Frequencies:", top_frequencies(words, 10))

        word_cloud(words, "Word Cloud")
    else:
        words = preprocess_text(opt)
        print("\nPreprocessed Words:", words)

        stemmed, lemmatized = stem_and_lemmatize(words)
        print("\nStemmed Words:", stemmed)
        print("\nLemmatized Words:", lemmatized)

        bigrams = generate_ngrams(words, 2)
        trigrams = generate_ngrams(words, 3)
        print("\nBigrams:", bigrams)
        print("\nTrigrams:", trigrams)

        pos_tags = pos_tagging(words)
        print("\nPOS Tags:", pos_tags)

        print("\nTop Word Frequencies:", top_frequencies(words, 10))

        word_cloud(words, "Word Cloud")

main()

