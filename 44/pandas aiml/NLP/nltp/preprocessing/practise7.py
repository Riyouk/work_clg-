from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import re
nltk.download('wordnet')
nltk.download('omw-1.4')
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()  

def tokanize_and_pos_tag(text):
    """Tokenize text and perform POS tagging"""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    cleaned_text = [word for word in text if word not in STOPWORDS]
    pos_tags = nltk.pos_tag(cleaned_text)
    return pos_tags 

def get_wordnet_pos(word):
    """Convert NLTK POS tag to WordNet POS tag"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if tag not found

def process_text_with_auto_pos(text):
    """Process text with automatic POS detection for lemmatization"""
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    for word, pos in pos_tags:
        wn_pos = get_wordnet_pos(word)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        stem = ps.stem(word)
        print(f"{word} (POS: {wn_pos}): Lemmatized: {lemma} | Stemmed: {stem}")


def main():
    file_path = 'C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/text.txt'  # Replace with your file path
    output_path = 'C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/post.txt'  # Output file for POS tags

    text = read_file(file_path)
    print("Original Text:/n", text)

    pos_tags = tokanize_and_pos_tag(text)
    print("/nTokenized and POS Tagged:/n", pos_tags)

    # Write POS tags to file
    with open(output_path, 'w') as f:
        for word, tag in pos_tags:
            f.write(f"{word}\t{tag}\n")

    print(f"POS tags written to {output_path}")

    print("/nProcessing text with automatic POS detection:")
    process_text_with_auto_pos(text)

if __name__ == "__main__":  
    main()