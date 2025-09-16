import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import re

def download_nltk_resources():
    """
    Download required NLTK resources for text processing
    """
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

def initialize_nlp_tools():
    """
    Initialize NLP tools and load stopwords
    
    Returns:
        tuple: (Porter Stemmer, WordNet Lemmatizer, Stopwords set)
    """
    # Initialize Porter Stemmer for word stemming
    ps = PorterStemmer()
    
    # Initialize WordNet Lemmatizer for word lemmatization
    lemmatizer = WordNetLemmatizer()
    
    # Load English stopwords
    stopwords_set = set(stopwords.words('english'))
    
    print("NLP tools initialized successfully")
    return ps, lemmatizer, stopwords_set

def get_pos_descriptions_dictionary():
    """
    Create a comprehensive dictionary mapping POS tags to their descriptions
    
    Returns:
        dict: Dictionary with POS tag as key and description as value
    """
    pos_descriptions = {
        # Noun tags
        'NN': 'Noun, singular',
        'NNS': 'Noun, plural',
        'NNP': 'Proper noun, singular',
        'NNPS': 'Proper noun, plural',
        
        # Verb tags
        'VB': 'Verb, base form',
        'VBD': 'Verb, past tense',
        'VBG': 'Verb, gerund or present participle',
        'VBN': 'Verb, past participle',
        'VBP': 'Verb, non-3rd person singular present',
        'VBZ': 'Verb, 3rd person singular present',
        
        # Adjective tags
        'JJ': 'Adjective',
        'JJR': 'Adjective, comparative',
        'JJS': 'Adjective, superlative',
        
        # Adverb tags
        'RB': 'Adverb',
        'RBR': 'Adverb, comparative',
        'RBS': 'Adverb, superlative',
        
        # Pronoun tags
        'PRP': 'Personal pronoun',
        'PRP$': 'Possessive pronoun',
        'WP': 'Wh-pronoun',
        'WP$': 'Possessive wh-pronoun',
        
        # Determiner tags
        'DT': 'Determiner',
        'WDT': 'Wh-determiner',
        
        # Preposition and conjunction tags
        'IN': 'Preposition or subordinating conjunction',
        'CC': 'Coordinating conjunction',
        
        # Other tags
        'CD': 'Cardinal number',
        'EX': 'Existential there',
        'FW': 'Foreign word',
        'LS': 'List item marker',
        'MD': 'Modal',
        'PDT': 'Predeterminer',
        'POS': 'Possessive ending',
        'RP': 'Particle',
        'SYM': 'Symbol',
        'TO': 'to',
        'UH': 'Interjection',
        'WRB': 'Wh-adverb'
    }
    
    return pos_descriptions

def clean_text_from_punctuation(text):
    """
    Remove non-alphabetic characters from text and retain only words
    
    Args:
        text (str): Input text with punctuation and special characters
        
    Returns:
        str: Cleaned text with only alphabetic characters and spaces
    """
    # Remove all non-alphabetic characters using regex
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text)
    
    print("Text cleaned: punctuation and numbers removed")
    print(f"Cleaned text: {cleaned_text}")
    
    return cleaned_text

def convert_to_lowercase(text):
    """
    Convert text to lowercase for uniformity
    
    Args:
        text (str): Input text in any case
        
    Returns:
        str: Text converted to lowercase
    """
    lowercase_text = text.lower()
    
    print("Text converted to lowercase")
    print(f"Lowercase text: {lowercase_text}")
    
    return lowercase_text

def tokenize_text(text):
    """
    Split text into individual words (tokens)
    
    Args:
        text (str): Input text to be tokenized
        
    Returns:
        list: List of word tokens
    """
    # Split text by whitespace to create tokens
    tokens = text.split()
    
    print("Text tokenized")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    return tokens

def remove_stopwords_and_stem(tokens, stemmer, stopwords_set):
    """
    Remove stopwords and apply stemming to reduce words to their root form
    
    Args:
        tokens (list): List of word tokens
        stemmer (PorterStemmer): Porter stemmer object
        stopwords_set (set): Set of stopwords to remove
        
    Returns:
        list: List of stemmed words without stopwords
    """
    # Remove stopwords and apply stemming
    stemmed_words = [stemmer.stem(word) for word in tokens if word not in stopwords_set]
    
    print("Stopwords removed and stemming applied")
    print(f"Stemmed words: {stemmed_words}")
    print(f"Number of words after processing: {len(stemmed_words)}")
    
    return stemmed_words

def remove_stopwords_and_lemmatize(tokens, lemmatizer, stopwords_set):
    """
    Remove stopwords and apply lemmatization to get dictionary form of words
    
    Args:
        tokens (list): List of word tokens
        lemmatizer (WordNetLemmatizer): WordNet lemmatizer object
        stopwords_set (set): Set of stopwords to remove
        
    Returns:
        list: List of lemmatized words without stopwords
    """
    # Remove stopwords and apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
    
    print("Stopwords removed and lemmatization applied")
    print(f"Lemmatized words: {lemmatized_words}")
    print(f"Number of words after processing: {len(lemmatized_words)}")
    
    return lemmatized_words

def perform_pos_tagging(tokens):
    """
    Perform Part-of-Speech tagging on the given tokens
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of tuples containing (word, pos_tag)
    """
    # Apply POS tagging using NLTK
    pos_tags = nltk.pos_tag(tokens)
    
    print("POS tagging completed")
    print(f"POS tags: {pos_tags}")
    
    return pos_tags

def extract_pos_components(pos_tags):
    """
    Extract words and POS tags into separate lists
    
    Args:
        pos_tags (list): List of tuples containing (word, pos_tag)
        
    Returns:
        tuple: (words_list, pos_tags_list)
    """
    # Separate words and POS tags into different lists
    words_list = []
    pos_tags_list = []
    
    for word, pos in pos_tags:
        words_list.append(word)
        pos_tags_list.append(pos)
    
    print("Words and POS tags extracted into separate lists")
    print(f"Words: {words_list}")
    print(f"POS tags: {pos_tags_list}")
    
    return words_list, pos_tags_list

def get_pos_descriptions_for_tags(pos_tags_list, pos_descriptions_dict):
    """
    Get human-readable descriptions for POS tags
    
    Args:
        pos_tags_list (list): List of POS tags
        pos_descriptions_dict (dict): Dictionary mapping POS tags to descriptions
        
    Returns:
        list: List of POS descriptions
    """
    descriptions_list = []
    
    # Get description for each POS tag
    for pos_tag in pos_tags_list:
        description = pos_descriptions_dict.get(pos_tag, 'Unknown POS tag')
        descriptions_list.append(description)
    
    print("POS descriptions generated")
    print(f"Descriptions: {descriptions_list}")
    
    return descriptions_list

def generate_ngrams(text_tokens, n_values=[1, 2, 3]):
    """
    Generate n-grams (unigrams, bigrams, trigrams) for each word
    
    Args:
        text_tokens (list): List of processed text tokens
        n_values (list): List of n values for n-gram generation
        
    Returns:
        dict: Dictionary containing n-grams for each n value
    """
    ngram_results = {}
    
    # Generate n-grams for each word in the text
    for n in n_values:
        ngram_results[f'{n}-gram'] = []
        
        for word in text_tokens:
            # Generate n-grams for individual words (character level)
            word_ngrams = list(ngrams(word, n))
            ngram_results[f'{n}-gram'].extend(word_ngrams)
    
    print("N-grams generated")
    for key, value in ngram_results.items():
        print(f"{key}: {value[:10]}...")  # Show first 10 n-grams
    
    return ngram_results

def save_results_to_csv(words_list, pos_tags_list, descriptions_list, output_path):
    """
    Save the processing results to a CSV file
    
    Args:
        words_list (list): List of processed words
        pos_tags_list (list): List of POS tags
        descriptions_list (list): List of POS descriptions
        output_path (str): Path where CSV file will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            # Write CSV header
            f.write("words,pos_key,pos_description\\n")
            
            # Write data rows
            for word, pos_tag, description in zip(words_list, pos_tags_list, descriptions_list):
                f.write(f"{word},{pos_tag},{description}\\n")
        
        print(f"Results successfully saved to: {output_path}")
        print(f"Total records saved: {len(words_list)}")
        return True
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

def display_processing_summary(original_text, words_list, pos_tags_list):
    """
    Display a summary of the text processing results
    
    Args:
        original_text (str): Original input text
        words_list (list): List of processed words
        pos_tags_list (list): List of POS tags
    """
    print("\\n" + "="*60)
    print("TEXT PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Original text length: {len(original_text)} characters")
    print(f"Number of words after processing: {len(words_list)}")
    print(f"Number of unique POS tags: {len(set(pos_tags_list))}")
    
    # Count POS tag frequencies
    pos_counts = {}
    for pos in pos_tags_list:
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    print("\\nPOS Tag Frequencies:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {count}")
    
    print("="*60)

def main():
    """
    Main function to orchestrate the entire NLP text processing pipeline
    """
    print("Starting NLP Text Processing Pipeline")
    print("="*50)
    
    # Input text
    sentence = "!Lorem ipsum dolor, sit amet consectetur adipisicing elit. Minus dicta nemo, labore consequatur dolorum odit maxime sequi neque at nesciunt sunt repellendus perspiciatis cupiditate, totam sed provident et explicabo quaerat corrupti laudantium fugit dolores vitae! Eum tempora iste magnam excepturi? Animi reprehenderit libero aut molestias adipisci voluptas atque asperiores laboriosam debitis deserunt aspernatur beatae est itaque, hic numquam fugiat fugit cupiditate, veniam nam error quae voluptatibus facilis deleniti eius. Iusto laboriosam harum inventore laborum, id odit deserunt excepturi."
    
    print(f"Original text: {sentence}")
    print("\\n" + "-"*50)
    
    # Step 1: Download NLTK resources
    print("Step 1: Downloading NLTK resources...")
    download_nltk_resources()
    
    # Step 2: Initialize NLP tools
    print("\\nStep 2: Initializing NLP tools...")
    stemmer, lemmatizer, stopwords_set = initialize_nlp_tools()
    
    # Step 3: Get POS descriptions dictionary
    print("\\nStep 3: Loading POS descriptions...")
    pos_descriptions_dict = get_pos_descriptions_dictionary()
    
    # Step 4: Clean text from punctuation
    print("\\nStep 4: Cleaning text from punctuation...")
    cleaned_text = clean_text_from_punctuation(sentence)
    
    # Step 5: Convert to lowercase
    print("\\nStep 5: Converting to lowercase...")
    lowercase_text = convert_to_lowercase(cleaned_text)
    
    # Step 6: Tokenize text
    print("\\nStep 6: Tokenizing text...")
    tokens = tokenize_text(lowercase_text)
    
    # Step 7: Remove stopwords and lemmatize (using lemmatization for final processing)
    print("\\nStep 7: Removing stopwords and applying lemmatization...")
    processed_tokens = remove_stopwords_and_lemmatize(tokens, lemmatizer, stopwords_set)
    
    # Optional: Show stemming results too
    print("\\nOptional: Showing stemming results...")
    stemmed_tokens = remove_stopwords_and_stem(tokens, stemmer, stopwords_set)
    
    # Step 8: Perform POS tagging
    print("\\nStep 8: Performing POS tagging...")
    pos_tags = perform_pos_tagging(processed_tokens)
    
    # Step 9: Extract components
    print("\\nStep 9: Extracting words and POS tags...")
    words_list, pos_tags_list = extract_pos_components(pos_tags)
    
    # Step 10: Get POS descriptions
    print("\\nStep 10: Getting POS descriptions...")
    descriptions_list = get_pos_descriptions_for_tags(pos_tags_list, pos_descriptions_dict)
    
    # Step 11: Generate n-grams
    print("\\nStep 11: Generating n-grams...")
    ngram_results = generate_ngrams(processed_tokens)
    
    # Step 12: Save results to CSV
    output_path = "C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/manual/dataframe.csv"
    print(f"\\nStep 12: Saving results to CSV...")
    success = save_results_to_csv(words_list, pos_tags_list, descriptions_list, output_path)
    
    # Step 13: Display summary
    print("\\nStep 13: Displaying processing summary...")
    display_processing_summary(sentence, words_list, pos_tags_list)
    
    if success:
        print("\\nProcessing completed successfully!")
        print(f"Results saved to: {output_path}")
    else:
        print("\\nProcessing completed with some errors in file saving.")

# Run the main function when script is executed directly
if __name__ == "__main__":
    main()