from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

def download_nltk_resources():
    """
    Download all required NLTK resources for text processing
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
    Initialize Porter Stemmer, WordNet Lemmatizer, and stopwords
    Returns: tuple of (stemmer, lemmatizer, stopwords_set)
    """
    # Initialize Porter Stemmer for word stemming
    stemmer = PorterStemmer()
    
    # Initialize WordNet Lemmatizer for word lemmatization
    lemmatizer = WordNetLemmatizer()
    
    # Load English stopwords
    stopwords_set = set(stopwords.words('english'))
    
    return stemmer, lemmatizer, stopwords_set

def get_pos_description():
    """
    Create a dictionary mapping POS tags to their descriptions
    Returns: dictionary with POS tag as key and description as value
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

def read_text_file(file_path):
    """
    Read text content from a file
    
    Args:
        file_path (str): Path to the input text file
        
    Returns:
        str: Content of the file, or None if error occurs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"Successfully read file: {file_path}")
            return content
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def clean_and_tokenize_text(text, stopwords_set):
    """
    Clean text by removing non-alphabetic characters and stopwords, then tokenize
    
    Args:
        text (str): Raw text to be processed
        stopwords_set (set): Set of stopwords to remove
        
    Returns:
        list: List of cleaned tokens
    """
    # Remove all non-alphabetic characters and convert to lowercase
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase and split into words
    words = cleaned_text.lower().split()
    
    # Remove stopwords and words with length less than 2
    filtered_words = [word for word in words if word not in stopwords_set and len(word) > 1]
    
    print(f"Original words: {len(text.split())}, After cleaning: {len(filtered_words)}")
    
    return filtered_words

def perform_pos_tagging(tokens):
    """
    Perform Part-of-Speech tagging on tokens
    
    Args:
        tokens (list): List of tokens to be POS tagged
        
    Returns:
        list: List of tuples containing (word, pos_tag)
    """
    # Use NLTK's POS tagger to assign POS tags to each token
    pos_tags = nltk.pos_tag(tokens)
    
    print(f"POS tagging completed for {len(pos_tags)} tokens")
    
    return pos_tags

def create_pos_dataframe(pos_tags, pos_descriptions):
    """
    Create a pandas DataFrame with words, POS keys, and POS descriptions
    
    Args:
        pos_tags (list): List of tuples containing (word, pos_tag)
        pos_descriptions (dict): Dictionary mapping POS tags to descriptions
        
    Returns:
        pandas.DataFrame: DataFrame with columns 'word', 'pos_key', 'pos_description'
    """
    # Create lists for DataFrame columns
    words = []
    pos_keys = []
    pos_descriptions_list = []
    
    # Process each word-tag pair
    for word, pos_tag in pos_tags:
        words.append(word)
        pos_keys.append(pos_tag)
        
        # Get description for the POS tag, default to 'Unknown' if not found
        description = pos_descriptions.get(pos_tag, 'Unknown POS tag')
        pos_descriptions_list.append(description)
    
    # Create DataFrame
    df = pd.DataFrame({
        'word': words,
        'pos_key': pos_keys,
        'pos_description': pos_descriptions_list
    })
    
    print(f"DataFrame created with {len(df)} rows")
    
    return df

def write_dataframe_to_file(dataframe, output_path):
    """
    Write DataFrame to a tab-separated text file
    
    Args:
        dataframe (pandas.DataFrame): DataFrame to write
        output_path (str): Path for the output file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Write DataFrame to tab-separated file using manual method for reliability
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("word\tpos_key\tpos_description\n")
            
            # Write data rows
            for _, row in dataframe.iterrows():
                f.write(f"{row['word']}\t{row['pos_key']}\t{row['pos_description']}\n")
        
        print(f"Successfully wrote {len(dataframe)} rows to {output_path}")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

def display_sample_results(dataframe, num_samples=10):
    """
    Display sample results from the DataFrame
    
    Args:
        dataframe (pandas.DataFrame): DataFrame to display
        num_samples (int): Number of sample rows to display
    """
    print(f"\nSample results (first {num_samples} rows):")
    print("-" * 80)
    print(f"{'Word':<15} {'POS Key':<8} {'POS Description':<40}")
    print("-" * 80)
    
    for i in range(min(num_samples, len(dataframe))):
        row = dataframe.iloc[i]
        print(f"{row['word']:<15} {row['pos_key']:<8} {row['pos_description']:<40}")
    
    if len(dataframe) > num_samples:
        print("...")
    print("-" * 80)

def get_pos_statistics(dataframe):
    """
    Calculate and display statistics about POS tags
    
    Args:
        dataframe (pandas.DataFrame): DataFrame containing POS data
    """
    print("\nPOS Tag Statistics:")
    print("-" * 50)
    
    # Count frequency of each POS tag
    pos_counts = dataframe['pos_key'].value_counts()
    
    print(f"{'POS Tag':<8} {'Count':<8} {'Percentage':<12}")
    print("-" * 30)
    
    total_words = len(dataframe)
    for pos_tag, count in pos_counts.head(10).items():
        percentage = (count / total_words) * 100
        print(f"{pos_tag:<8} {count:<8} {percentage:<12.2f}%")
    
    print(f"\nTotal unique POS tags: {len(pos_counts)}")
    print(f"Total words processed: {total_words}")

def main():
    """
    Main function to orchestrate the entire NLP text processing pipeline
    """
    print("Starting NLP Text Processing Pipeline")
    print("=" * 50)
    
    # Step 1: Download required NLTK resources
    print("Step 1: Downloading NLTK resources...")
    download_nltk_resources()
    
    # Step 2: Initialize NLP tools
    print("\nStep 2: Initializing NLP tools...")
    stemmer, lemmatizer, stopwords_set = initialize_nlp_tools()
    
    # Step 3: Get POS descriptions dictionary
    print("Step 3: Loading POS descriptions...")
    pos_descriptions = get_pos_description()
    
    # Step 4: Define file paths
    input_file_path = 'C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/text.txt'  # Change this to your input file path
    output_file_path = 'C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/trying.txt'  # Output file with required columns
    
    # Step 5: Read input text file
    print(f"\nStep 4: Reading input file - {input_file_path}")
    text_content = read_text_file(input_file_path)
    
    if text_content is None:
        print("Failed to read input file. Exiting.")
        return
    
    print(f"Text preview: {text_content[:200]}...")
    
    # Step 6: Clean and tokenize text
    print("\nStep 5: Cleaning and tokenizing text...")
    tokens = clean_and_tokenize_text(text_content, stopwords_set)
    
    # Step 7: Perform POS tagging
    print("\nStep 6: Performing POS tagging...")
    pos_tags = perform_pos_tagging(tokens)
    
    # Step 8: Create DataFrame with required columns
    print("\nStep 7: Creating DataFrame with POS analysis...")
    pos_dataframe = create_pos_dataframe(pos_tags, pos_descriptions)
    
    # Step 9: Display sample results
    display_sample_results(pos_dataframe)
    
    # Step 10: Calculate and display statistics
    get_pos_statistics(pos_dataframe)
    
    # Step 11: Write results to output file
    print(f"\nStep 8: Writing results to {output_file_path}...")
    success = write_dataframe_to_file(pos_dataframe, output_file_path)
    
    if success:
        print("\nProcessing completed successfully!")
        print(f"Output file created: {output_file_path}")
        print("Columns: word, pos_key, pos_description")
    else:
        print("Failed to write output file.")

# Run the main function when script is executed directly
if __name__ == "__main__":
    main()