import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import re

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('puck_tab',quiet=True)
# nltk.download('averaged_perceptron_tagger', quiet=True)
# nltk.download('maxent_ne_chunker', quiet=True)
# nltk.download('words', quiet=True)

oper = input("Enter text to preprocess: ")

def preprocess_text(text):
    # Retain only words
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    # words = text.split()
    words = nltk.word_tokenize(text)
    
    # Remove stopwords
    STOPWORDS = set(stopwords.words('english'))
    words = [word for word in words if word not in STOPWORDS]
    
    return words

def stem_and_lemmatize(words):
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    stemmed_words = [ps.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    return stemmed_words, lemmatized_words

def generate_ngrams(words, n):
    return list(ngrams(words, n))

def post_tagging(words):
    return nltk.pos_tag(words)

def ner_chunking(pos_tags):
    return nltk.ne_chunk(pos_tags)

def main(text):
    words = preprocess_text(text)
    print("\nPreprocessed Words:", words)
    
    stemmed_words, lemmatized_words = stem_and_lemmatize(words)
    print("\nStemmed Words:", stemmed_words)
    print("\nLemmatized Words:", lemmatized_words)
    
    bigrams = generate_ngrams(words, 2)
    trigrams = generate_ngrams(words, 3)
    print("\nBigrams:", bigrams)
    print("\nTrigrams:", trigrams)
    
    pos_tags = post_tagging(words)
    print("\nPOS Tags:", pos_tags)

    named_entities = ner_chunking(pos_tags)
    print("\nNamed Entities:", named_entities)

if __name__ == "__main__":
    main(oper)