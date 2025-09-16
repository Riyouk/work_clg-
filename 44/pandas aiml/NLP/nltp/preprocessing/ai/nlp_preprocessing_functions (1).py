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

def read_pos_descriptions_from_csv(csv_file_path):
    """
    Read POS descriptions from CSV file and create a dictionary
    
    Args:
        csv_file_path (str): Path to the CSV file containing POS tags and descriptions
        
    Returns:
        dict: Dictionary mapping POS tags to their descriptions
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read CSV file: {csv_file_path}")
        print(f"CSV file shape: {df.shape}")
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Create dictionary from CSV data
        # Assuming the CSV has columns like 'pos_key' and 'pos_description' or similar
        pos_descriptions = {}
        
        # Try different possible column names
        pos_col = None
        desc_col = None
        
        for col in df.columns:
            if 'pos' in col.lower() and 'key' in col.lower():
                pos_col = col
            elif 'pos' in col.lower() and ('desc' in col.lower() or 'description' in col.lower()):
                desc_col = col
            elif col.lower() in ['pos_tag', 'tag', 'pos']:
                pos_col = col
            elif col.lower() in ['description', 'desc', 'meaning']:
                desc_col = col
        
        # If standard columns not found, use first two columns
        if pos_col is None or desc_col is None:
            columns = df.columns.tolist()
            pos_col = columns[1] if len(columns) > 1 else columns[0]  # Assume second column is POS
            desc_col = columns[2] if len(columns) > 2 else columns[0]  # Assume third column is description
        
        print(f"Using POS column: {pos_col}")
        print(f"Using description column: {desc_col}")
        
        # Create the dictionary
        for index, row in df.iterrows():
            pos_tag = str(row[pos_col]).strip()
            description = str(row[desc_col]).strip()
            pos_descriptions[pos_tag] = description
        
        print(f"Loaded {len(pos_descriptions)} POS descriptions from CSV")
        return pos_descriptions
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Using default POS descriptions...")
        # Fallback to default descriptions
        return get_default_pos_descriptions()

def get_default_pos_descriptions():
    """
    Get default POS descriptions as fallback
    
    Returns:
        dict: Default POS descriptions dictionary
    """
    pos_descriptions = {
        'NN': 'Noun, singular', 'NNS': 'Noun, plural', 'NNP': 'Proper noun, singular',
        'NNPS': 'Proper noun, plural', 'VB': 'Verb, base form', 'VBD': 'Verb, past tense',
        'VBG': 'Verb, gerund or present participle', 'VBN': 'Verb, past participle',
        'VBP': 'Verb, non-3rd person singular present', 'VBZ': 'Verb, 3rd person singular present',
        'JJ': 'Adjective', 'JJR': 'Adjective, comparative', 'JJS': 'Adjective, superlative',
        'RB': 'Adverb', 'RBR': 'Adverb, comparative', 'RBS': 'Adverb, superlative',
        'PRP': 'Personal pronoun', 'PRP$': 'Possessive pronoun', 'DT': 'Determiner',
        'IN': 'Preposition or subordinating conjunction', 'CC': 'Coordinating conjunction',
        'CD': 'Cardinal number', 'WP': 'Wh-pronoun', 'WDT': 'Wh-determiner',
        'MD': 'Modal', 'TO': 'to', 'WRB': 'Wh-adverb'
    }
    return pos_descriptions

def read_input_text_file(file_path):
    """
    Read text content from input file
    
    Args:
        file_path (str): Path to the input text file
        
    Returns:
        str: Content of the file, or None if error occurs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"Successfully read input file: {file_path}")
            print(f"Text length: {len(content)} characters")
            return content
    except FileNotFoundError:
        print(f"Error: Input file not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None

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
    
    print(f"Text tokenized: {len(tokens)} tokens created")
    return tokens

def remove_stopwords(tokens, stopwords_set):
    """
    Remove stopwords from tokens
    
    Args:
        tokens (list): List of word tokens
        stopwords_set (set): Set of stopwords to remove
        
    Returns:
        list: List of tokens without stopwords
    """
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords_set and len(word) > 1]
    
    print(f"Stopwords removed: {len(filtered_tokens)} tokens remaining")
    return filtered_tokens

def apply_stemming_and_lemmatization(tokens, stemmer, lemmatizer):
    """
    Apply both stemming and lemmatization to tokens
    
    Args:
        tokens (list): List of word tokens
        stemmer (PorterStemmer): Porter stemmer object
        lemmatizer (WordNetLemmatizer): WordNet lemmatizer object
        
    Returns:
        tuple: (stemmed_words, lemmatized_words)
    """
    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in tokens]
    
    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    
    print("Stemming and lemmatization applied")
    return stemmed_words, lemmatized_words

def perform_pos_tagging_for_all_forms(original_tokens, stemmed_tokens, lemmatized_tokens):
    """
    Perform POS tagging for original, stemmed, and lemmatized tokens
    
    Args:
        original_tokens (list): Original tokens
        stemmed_tokens (list): Stemmed tokens
        lemmatized_tokens (list): Lemmatized tokens
        
    Returns:
        tuple: (original_pos, stemmed_pos, lemmatized_pos)
    """
    # Apply POS tagging to all forms
    original_pos = nltk.pos_tag(original_tokens)
    stemmed_pos = nltk.pos_tag(stemmed_tokens)
    lemmatized_pos = nltk.pos_tag(lemmatized_tokens)
    
    print("POS tagging completed for all word forms")
    return original_pos, stemmed_pos, lemmatized_pos

def get_unique_words(tokens):
    """
    Get unique words from tokens while preserving order
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        list: List of unique tokens in order of first appearance
    """
    seen = set()
    unique_tokens = []
    
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    print(f"Unique words extracted: {len(unique_tokens)} unique words from {len(tokens)} total")
    return unique_tokens

def create_comprehensive_dataframe(original_tokens, stemmed_tokens, lemmatized_tokens, 
                                 original_pos, stemmed_pos, lemmatized_pos, pos_descriptions_dict):
    """
    Create a comprehensive DataFrame with all required columns
    
    Args:
        original_tokens (list): Original tokens
        stemmed_tokens (list): Stemmed tokens  
        lemmatized_tokens (list): Lemmatized tokens
        original_pos (list): POS tags for original tokens
        stemmed_pos (list): POS tags for stemmed tokens
        lemmatized_pos (list): POS tags for lemmatized tokens
        pos_descriptions_dict (dict): Dictionary mapping POS tags to descriptions
        
    Returns:
        pandas.DataFrame: DataFrame with all required columns
    """
    # Get unique words
    unique_words = get_unique_words(original_tokens)
    
    # Prepare data for DataFrame
    data_rows = []
    
    for i, original_word in enumerate(original_tokens):
        # Get corresponding stemmed and lemmatized words
        stemmed_word = stemmed_tokens[i] if i < len(stemmed_tokens) else ""
        lemmatized_word = lemmatized_tokens[i] if i < len(lemmatized_tokens) else ""
        
        # Get POS tags
        original_pos_tag = original_pos[i][1] if i < len(original_pos) else ""
        stemmed_pos_tag = stemmed_pos[i][1] if i < len(stemmed_pos) else ""
        lemmatized_pos_tag = lemmatized_pos[i][1] if i < len(lemmatized_pos) else ""
        
        # Get descriptions
        original_desc = pos_descriptions_dict.get(original_pos_tag, 'Unknown POS tag')
        stemmed_desc = pos_descriptions_dict.get(stemmed_pos_tag, 'Unknown POS tag')
        lemmatized_desc = pos_descriptions_dict.get(lemmatized_pos_tag, 'Unknown POS tag')
        
        # Check if word is unique (first occurrence)
        is_unique = original_word in unique_words
        if is_unique:
            unique_words.remove(original_word)  # Remove to avoid duplicates
        
        row_data = {
            'word': original_word,
            'pos_key': original_pos_tag,
            'pos_description': original_desc,
            'unique_word': original_word if is_unique else '',
            'stemmed_word': stemmed_word,
            'stem_pos_tag': stemmed_pos_tag,
            'stem_pos_description': stemmed_desc,
            'lemmatized_word': lemmatized_word,
            'lem_pos_tag': lemmatized_pos_tag,
            'lem_pos_description': lemmatized_desc
        }
        
        data_rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    print(f"Comprehensive DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def save_comprehensive_results_to_csv(dataframe, output_path):
    """
    Save the comprehensive processing results to a CSV file
    
    Args:
        dataframe (pandas.DataFrame): DataFrame with all processing results
        output_path (str): Path where CSV file will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Save DataFrame to CSV
        dataframe.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Comprehensive results successfully saved to: {output_path}")
        print(f"Total records saved: {len(dataframe)}")
        print(f"Columns saved: {list(dataframe.columns)}")
        return True
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

def display_processing_summary(dataframe):
    """
    Display a summary of the comprehensive text processing results
    
    Args:
        dataframe (pandas.DataFrame): DataFrame with processing results
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE TEXT PROCESSING SUMMARY")
    print("="*80)
    
    print(f"Total words processed: {len(dataframe)}")
    print(f"Unique words: {len(dataframe[dataframe['unique_word'] != ''])}")
    
    # POS tag statistics for original words
    pos_counts = dataframe['pos_key'].value_counts()
    print(f"Number of unique POS tags: {len(pos_counts)}")
    
    print("\nTop 10 POS Tag Frequencies:")
    for pos, count in pos_counts.head(10).items():
        percentage = (count / len(dataframe)) * 100
        print(f"  {pos}: {count} ({percentage:.2f}%)")
    
    print("\nSample of processed data:")
    print(dataframe.head(5).to_string())
    
    print("="*80)

def main():
    """
    Main function to orchestrate the entire enhanced NLP text processing pipeline
    """
    print("Starting Enhanced NLP Text Processing Pipeline")
    print("="*60)
    
    # File paths - modify these according to your requirements
    pos_csv_file = "C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/pos_tags.csv"
    input_text_file = "C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/manual/text.txt"  # Change this to your input text file
    output_csv_file = "C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/ai/comprehensive_nlp_results.csv"  # Output file
    
    # Step 1: Download NLTK resources
    print("Step 1: Downloading NLTK resources...")
    download_nltk_resources()
    
    # Step 2: Initialize NLP tools
    print("\nStep 2: Initializing NLP tools...")
    stemmer, lemmatizer, stopwords_set = initialize_nlp_tools()
    
    # Step 3: Read POS descriptions from CSV file
    print(f"\nStep 3: Reading POS descriptions from CSV file...")
    pos_descriptions_dict = read_pos_descriptions_from_csv(pos_csv_file)
    
    # Step 4: Read input text file
    print(f"\nStep 4: Reading input text file...")
    if input_text_file == "input_text.txt":
        # Use sample text if no specific file provided
        text_content = "!Lorem ipsum dolor, sit amet consectetur adipisicing elit. Minus dicta nemo, labore consequatur dolorum odit maxime sequi neque at nesciunt sunt repellendus perspiciatis cupiditate, totam sed provident et explicabo quaerat corrupti laudantium fugit dolores vitae! Eum tempora iste magnam excepturi? Animi reprehenderit libero aut molestias adipisci voluptas atque asperiores laboriosam debitis deserunt aspernatur beatae est itaque, hic numquam fugiat fugit cupiditate, veniam nam error quae voluptatibus facilis deleniti eius. Iusto laboriosam harum inventore laborum, id odit deserunt excepturi."
        print("Using sample text (no input file specified)")
    else:
        text_content = read_input_text_file(input_text_file)
        if text_content is None:
            print("Failed to read input file. Exiting.")
            return
    
    print(f"Input text preview: {text_content[:200]}...")
    
    # Step 5: Clean text from punctuation
    print("\nStep 5: Cleaning text from punctuation...")
    cleaned_text = clean_text_from_punctuation(text_content)
    
    # Step 6: Convert to lowercase
    print("\nStep 6: Converting to lowercase...")
    lowercase_text = convert_to_lowercase(cleaned_text)
    
    # Step 7: Tokenize text
    print("\nStep 7: Tokenizing text...")
    all_tokens = tokenize_text(lowercase_text)
    
    # Step 8: Remove stopwords
    print("\nStep 8: Removing stopwords...")
    filtered_tokens = remove_stopwords(all_tokens, stopwords_set)
    
    # Step 9: Apply stemming and lemmatization
    print("\nStep 9: Applying stemming and lemmatization...")
    stemmed_tokens, lemmatized_tokens = apply_stemming_and_lemmatization(
        filtered_tokens, stemmer, lemmatizer)
    
    # Step 10: Perform POS tagging for all forms
    print("\nStep 10: Performing POS tagging for all word forms...")
    original_pos, stemmed_pos, lemmatized_pos = perform_pos_tagging_for_all_forms(
        filtered_tokens, stemmed_tokens, lemmatized_tokens)
    
    # Step 11: Create comprehensive DataFrame
    print("\nStep 11: Creating comprehensive DataFrame...")
    results_df = create_comprehensive_dataframe(
        filtered_tokens, stemmed_tokens, lemmatized_tokens,
        original_pos, stemmed_pos, lemmatized_pos, pos_descriptions_dict)
    
    # Step 12: Save results to CSV
    print(f"\nStep 12: Saving comprehensive results to CSV...")
    success = save_comprehensive_results_to_csv(results_df, output_csv_file)
    
    # Step 13: Display summary
    print("\nStep 13: Displaying processing summary...")
    display_processing_summary(results_df)
    
    if success:
        print(f"\nProcessing completed successfully!")
        print(f"Comprehensive results saved to: {output_csv_file}")
        print("Columns included: word, pos_key, pos_description, unique_word, stemmed_word, stem_pos_tag, stem_pos_description, lemmatized_word, lem_pos_tag, lem_pos_description")
    else:
        print("\nProcessing completed with some errors in file saving.")

# Run the main function when script is executed directly
if __name__ == "__main__":
    main()