import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams, FreqDist
import re
import string
from collections import Counter
import seaborn as sns

# Download necessary NLTK resources
resources = ['stopwords', 'punkt', 'wordnet']
for resource in resources:
    try:
        nltk.download(resource, quiet=True)
    except:
        print(f"Failed to download {resource}")

# Set of English stopwords
STOPWORDS = set(stopwords.words('english'))

# Sample text for processing
sample_text = """Lorem ipsum dolor, sit amet consectetur adipisicing elit. Minus dicta nemo, labore consequatur dolorum odit maxime sequi neque at nesciunt sunt repellendus perspiciatis cupiditate, totam sed provident et explicabo quaerat corrupti laudantium fugit dolores vitae! Eum tempora iste magnam excepturi? Animi reprehenderit libero aut molestias adipisci voluptas atque asperiores laboriosam debitis deserunt aspernatur beatae est itaque, hic numquam fugiat fugit cupiditate, veniam nam error quae voluptatibus facilis deleniti eius. Iusto laboriosam harum inventore laborum, id odit deserunt excepturi."""

# Initialize stemmer and lemmatizer
porter_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, remove_stopwords=True, stem=False, lemmatize=False):
    """Preprocess text by cleaning, tokenizing, and optionally removing stopwords and stemming/lemmatizing
    
    Args:
        text (str): Input text to process
        remove_stopwords (bool): Whether to remove stopwords
        stem (bool): Whether to apply stemming
        lemmatize (bool): Whether to apply lemmatization
        
    Returns:
        list: Processed tokens
    """
    # Remove special characters and numbers, keep only letters
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    
    # Apply stemming if requested
    if stem:
        tokens = [porter_stemmer.stem(token) for token in tokens]
    
    # Apply lemmatization if requested
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

def generate_ngrams(tokens, n=1):
    """Generate n-grams from tokens
    
    Args:
        tokens (list): List of tokens
        n (int): Size of n-grams to generate
        
    Returns:
        list: List of n-grams
    """
    return list(ngrams(tokens, n))

def analyze_text(text):
    """Perform comprehensive text analysis
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Sentence tokenization
    sentences = sent_tokenize(text)
    
    # Basic preprocessing (with stopwords)
    tokens_with_stopwords = preprocess_text(text, remove_stopwords=False)
    
    # Preprocessing (without stopwords)
    tokens = preprocess_text(text, remove_stopwords=True)
    
    # Stemming
    stemmed_tokens = preprocess_text(text, remove_stopwords=True, stem=True)
    
    # Lemmatization
    lemmatized_tokens = preprocess_text(text, remove_stopwords=True, lemmatize=True)
    
    # Generate n-grams
    unigrams = generate_ngrams(tokens, 1)
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)
    
    # Word frequency
    word_freq = Counter(tokens)
    
    return {
        'sentences': sentences,
        'tokens_with_stopwords': tokens_with_stopwords,
        'tokens': tokens,
        'stemmed_tokens': stemmed_tokens,
        'lemmatized_tokens': lemmatized_tokens,
        'unigrams': unigrams,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'word_freq': word_freq
    }

def visualize_word_frequency(word_freq, top_n=10):
    """Visualize word frequency
    
    Args:
        word_freq (Counter): Word frequency counter
        top_n (int): Number of top words to display
    """
    # Get top N words
    top_words = dict(word_freq.most_common(top_n))
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(top_words.keys(), top_words.values())
    plt.title(f'Top {top_n} Words Frequency')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compare_processing_methods(text):
    """Compare different text processing methods
    
    Args:
        text (str): Input text to process
    """
    # Original tokens (lowercase, no stopwords)
    tokens = preprocess_text(text, remove_stopwords=True)
    
    # Stemmed tokens
    stemmed_tokens = preprocess_text(text, remove_stopwords=True, stem=True)
    
    # Lemmatized tokens
    lemmatized_tokens = preprocess_text(text, remove_stopwords=True, lemmatize=True)
    
    # Print comparison for first 10 words
    print("\nComparison of processing methods (first 10 words):")
    comparison = pd.DataFrame({
        'Original': tokens[:10],
        'Stemmed': stemmed_tokens[:10],
        'Lemmatized': lemmatized_tokens[:10]
    })
    print(comparison)

# Main execution
if __name__ == "__main__":
    print("\n===== TEXT ANALYSIS DEMO =====\n")
    
    # Display original text
    print("Original Text:")
    print(sample_text)
    
    # Analyze text
    analysis_results = analyze_text(sample_text)
    
    # Display basic preprocessing results
    print("\n===== BASIC PREPROCESSING =====")
    print("\nText after removing special characters:")
    print(re.sub('[^a-zA-Z]', ' ', sample_text))
    
    print("\nText after converting to lowercase:")
    print(re.sub('[^a-zA-Z]', ' ', sample_text).lower())
    
    print("\nTokenized text:")
    print(analysis_results['tokens_with_stopwords'][:20], "...")
    
    # Display stopwords removal
    print("\n===== STOPWORDS REMOVAL =====")
    print(f"Number of tokens with stopwords: {len(analysis_results['tokens_with_stopwords'])}")
    print(f"Number of tokens without stopwords: {len(analysis_results['tokens'])}")
    print("\nTokens after removing stopwords:")
    print(analysis_results['tokens'][:20], "...")
    
    # Display stemming results
    print("\n===== STEMMING =====")
    print("\nTokens after stemming:")
    print(analysis_results['stemmed_tokens'][:20], "...")
    
    # Display lemmatization results
    print("\n===== LEMMATIZATION =====")
    print("\nTokens after lemmatization:")
    print(analysis_results['lemmatized_tokens'][:20], "...")
    
    # Compare processing methods
    compare_processing_methods(sample_text)
    
    # Display n-grams
    print("\n===== N-GRAMS =====")
    print("\nUnigrams (first 5):")
    print(analysis_results['unigrams'][:5])
    
    print("\nBigrams (first 5):")
    print(analysis_results['bigrams'][:5])
    
    print("\nTrigrams (first 5):")
    print(analysis_results['trigrams'][:5])
    
    # Display word frequency
    print("\n===== WORD FREQUENCY =====")
    print("\nTop 10 most frequent words:")
    for word, count in analysis_results['word_freq'].most_common(10):
        print(f"{word}: {count}")
    
    # Visualize word frequency (uncomment to display)
    # visualize_word_frequency(analysis_results['word_freq'])
    
    print("\n===== ANALYSIS COMPLETE =====")