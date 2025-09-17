import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams, ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import re
from collections import Counter

def download_nltk_resources():
    """
    Download all required NLTK resources for NER processing
    """
    required_resources = [
        'stopwords', 'wordnet', 'omw-1.4', 'punkt', 
        'averaged_perceptron_tagger', 'maxent_ne_chunker', 
        'words', 'vader_lexicon'
    ]
    
    print("Downloading NLTK resources for NER...")
    for resource in required_resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")
    
    print("NLTK resources download completed")

def initialize_nlp_tools():
    """
    Initialize NLP tools for text processing
    
    Returns:
        tuple: (PorterStemmer, WordNetLemmatizer, stopwords_set)
    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    
    print("NLP tools initialized successfully")
    return stemmer, lemmatizer, stopwords_set

def preprocess_text_for_ner(text, remove_stopwords=False):
    """
    Preprocess text specifically for Named Entity Recognition
    
    Args:
        text (str): Input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords (default: False for better NER)
        
    Returns:
        list: List of preprocessed tokens
    """
    # Clean text but preserve proper nouns and capitalization for better NER
    # Only remove non-alphabetic characters but keep spaces
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize using NLTK's word_tokenize for better sentence handling
    tokens = word_tokenize(cleaned_text)
    
    # Option to remove stopwords (usually not recommended for NER)
    if remove_stopwords:
        STOPWORDS = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in STOPWORDS and len(word) > 1]
    
    print(f"Text preprocessed: {len(tokens)} tokens extracted")
    return tokens

def perform_pos_tagging(tokens):
    """
    Perform Part-of-Speech tagging on tokens
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of (word, POS_tag) tuples
    """
    pos_tags = pos_tag(tokens)
    
    print(f"POS tagging completed for {len(pos_tags)} tokens")
    return pos_tags

def extract_named_entities(pos_tagged_tokens):
    """
    Extract named entities using NLTK's NER chunker
    
    Args:
        pos_tagged_tokens (list): List of (word, POS_tag) tuples
        
    Returns:
        tuple: (chunked_tree, extracted_entities)
    """
    # Perform named entity chunking
    chunked_tree = ne_chunk(pos_tagged_tokens)
    
    # Extract entities from the tree
    entities = []
    current_entity = []
    current_label = None
    
    for chunk in chunked_tree:
        if isinstance(chunk, Tree):
            # This is a named entity
            entity_name = ' '.join([token for token, pos in chunk.leaves()])
            entity_label = chunk.label()
            entities.append((entity_name, entity_label))
        else:
            # This is a regular word, not part of a named entity
            if current_entity:
                # Close any ongoing entity
                current_entity = []
                current_label = None
    
    print(f"Named entity extraction completed: {len(entities)} entities found")
    return chunked_tree, entities

def categorize_entities(entities):
    """
    Categorize and count named entities by type
    
    Args:
        entities (list): List of (entity, label) tuples
        
    Returns:
        dict: Dictionary with entity categories and their counts
    """
    entity_categories = {
        'PERSON': [],
        'ORGANIZATION': [],
        'GPE': [],  # Geopolitical entity (countries, cities, states)
        'LOCATION': [],
        'DATE': [],
        'TIME': [],
        'MONEY': [],
        'PERCENT': [],
        'FACILITY': [],
        'OTHER': []
    }
    
    # Categorize entities
    for entity_name, entity_label in entities:
        if entity_label in entity_categories:
            entity_categories[entity_label].append(entity_name)
        else:
            entity_categories['OTHER'].append(f"{entity_name} ({entity_label})")
    
    # Count entities by category
    entity_counts = {category: len(entities_list) 
                    for category, entities_list in entity_categories.items()}
    
    print("Entity categorization completed")
    return entity_categories, entity_counts

def generate_ngrams_for_entities(tokens, n_values=[2, 3]):
    """
    Generate n-grams which can help identify multi-word entities
    
    Args:
        tokens (list): List of tokens
        n_values (list): List of n-gram sizes to generate
        
    Returns:
        dict: Dictionary of n-grams by size
    """
    ngram_results = {}
    
    for n in n_values:
        ngrams_list = list(ngrams(tokens, n))
        # Convert tuples to strings for better readability
        ngrams_strings = [' '.join(gram) for gram in ngrams_list]
        ngram_results[f'{n}-grams'] = ngrams_strings
    
    print(f"N-grams generated: {[f'{n}-grams: {len(ngram_results[f\"{n}-grams\"])}' for n in n_values]}")
    return ngram_results

def apply_stemming_and_lemmatization(tokens, stemmer, lemmatizer):
    """
    Apply stemming and lemmatization to tokens
    
    Args:
        tokens (list): List of tokens
        stemmer (PorterStemmer): Stemmer object
        lemmatizer (WordNetLemmatizer): Lemmatizer object
        
    Returns:
        tuple: (stemmed_words, lemmatized_words)
    """
    stemmed_words = [stemmer.stem(token) for token in tokens]
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    
    print(f"Stemming and lemmatization applied to {len(tokens)} tokens")
    return stemmed_words, lemmatized_words

def create_ner_dataframe(tokens, pos_tags, entities, stemmed_words, lemmatized_words):
    """
    Create a comprehensive DataFrame with NER results
    
    Args:
        tokens (list): Original tokens
        pos_tags (list): POS tagged tokens
        entities (list): Named entities
        stemmed_words (list): Stemmed tokens
        lemmatized_words (list): Lemmatized tokens
        
    Returns:
        pandas.DataFrame: DataFrame with NER analysis
    """
    # Create entity lookup for quick reference
    entity_dict = {entity: label for entity, label in entities}
    
    # Prepare data rows
    data_rows = []
    
    for i, (word, pos_tag) in enumerate(pos_tags):
        # Check if this word is part of any named entity
        entity_label = 'O'  # Default: Outside entity
        
        # Simple entity matching (could be improved with more sophisticated matching)
        for entity_name, ent_label in entities:
            if word.lower() in entity_name.lower():
                entity_label = ent_label
                break
        
        row_data = {
            'word': word,
            'pos_tag': pos_tag,
            'entity_label': entity_label,
            'stemmed_word': stemmed_words[i] if i < len(stemmed_words) else '',
            'lemmatized_word': lemmatized_words[i] if i < len(lemmatized_words) else '',
            'is_entity': entity_label != 'O'
        }
        
        data_rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    print(f"NER DataFrame created with {len(df)} rows")
    return df

def save_ner_results(dataframe, entities, entity_categories, output_file):
    """
    Save NER results to CSV file and display summary
    
    Args:
        dataframe (pandas.DataFrame): NER results DataFrame
        entities (list): List of named entities
        entity_categories (dict): Categorized entities
        output_file (str): Output file path
        
    Returns:
        bool: Success status
    """
    try:
        # Save main DataFrame
        dataframe.to_csv(output_file, index=False, encoding='utf-8')
        
        # Create summary file
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("NAMED ENTITY RECOGNITION SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Total tokens processed: {len(dataframe)}\\n")
            f.write(f"Total named entities found: {len(entities)}\\n")
            f.write(f"Tokens identified as entities: {len(dataframe[dataframe['is_entity'] == True])}\\n\\n")
            
            f.write("ENTITIES BY CATEGORY:\\n")
            f.write("-" * 30 + "\\n")
            
            for category, entities_list in entity_categories.items():
                if entities_list:
                    f.write(f"\\n{category} ({len(entities_list)}):)\\n")
                    for entity in entities_list:
                        f.write(f"  - {entity}\\n")
        
        print(f"NER results saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        return True
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def display_ner_results(entities, entity_categories, entity_counts, chunked_tree):
    """
    Display comprehensive NER results
    
    Args:
        entities (list): List of named entities
        entity_categories (dict): Categorized entities
        entity_counts (dict): Entity count by category
        chunked_tree: NLTK chunked tree
    """
    print("\\n" + "=" * 70)
    print("NAMED ENTITY RECOGNITION RESULTS")
    print("=" * 70)
    
    print(f"\\nTotal Named Entities Found: {len(entities)}")
    
    if entities:
        print("\\nAll Named Entities:")
        print("-" * 40)
        for i, (entity, label) in enumerate(entities, 1):
            print(f"{i:2d}. {entity:<25} [{label}]")
    
    print("\\nEntity Categories Summary:")
    print("-" * 40)
    for category, count in entity_counts.items():
        if count > 0:
            print(f"{category:<15}: {count}")
    
    print("\\nDetailed Entity Categories:")
    print("-" * 40)
    for category, entities_list in entity_categories.items():
        if entities_list:
            print(f"\\n{category}:")
            for entity in entities_list:
                print(f"  - {entity}")
    
    print("\\n" + "=" * 70)

def process_user_input():
    """
    Get and validate user input
    
    Returns:
        str: User input text
    """
    while True:
        user_text = input("\\nEnter text for Named Entity Recognition: ").strip()
        
        if len(user_text) < 5:
            print("Please enter at least 5 characters for meaningful NER analysis.")
            continue
        
        return user_text

def main():
    """
    Main function to orchestrate the NER processing pipeline
    """
    print("Named Entity Recognition (NER) Text Processor")
    print("=" * 50)
    
    # Step 1: Download required resources
    download_nltk_resources()
    
    # Step 2: Initialize NLP tools
    stemmer, lemmatizer, stopwords_set = initialize_nlp_tools()
    
    # Step 3: Get user input
    text_input = process_user_input()
    
    print(f"\\nProcessing text: '{text_input[:100]}{'...' if len(text_input) > 100 else ''}'")
    print("-" * 50)
    
    # Step 4: Preprocess text (keeping capitalization for better NER)
    print("\\nStep 1: Preprocessing text...")
    tokens = preprocess_text_for_ner(text_input, remove_stopwords=False)
    print(f"Tokens: {tokens}")
    
    # Step 5: Apply stemming and lemmatization
    print("\\nStep 2: Applying stemming and lemmatization...")
    stemmed_words, lemmatized_words = apply_stemming_and_lemmatization(
        tokens, stemmer, lemmatizer)
    print(f"Stemmed: {stemmed_words}")
    print(f"Lemmatized: {lemmatized_words}")
    
    # Step 6: POS tagging
    print("\\nStep 3: Performing POS tagging...")
    pos_tags = perform_pos_tagging(tokens)
    print(f"POS Tags: {pos_tags}")
    
    # Step 7: Named Entity Recognition
    print("\\nStep 4: Extracting Named Entities...")
    chunked_tree, entities = extract_named_entities(pos_tags)
    
    # Step 8: Categorize entities
    print("\\nStep 5: Categorizing entities...")
    entity_categories, entity_counts = categorize_entities(entities)
    
    # Step 9: Generate n-grams
    print("\\nStep 6: Generating n-grams...")
    ngram_results = generate_ngrams_for_entities(tokens)
    print(f"Bigrams (first 5): {ngram_results['2-grams'][:5]}")
    print(f"Trigrams (first 5): {ngram_results['3-grams'][:5]}")
    
    # Step 10: Create comprehensive DataFrame
    print("\\nStep 7: Creating NER DataFrame...")
    ner_df = create_ner_dataframe(tokens, pos_tags, entities, stemmed_words, lemmatized_words)
    
    # Step 11: Display results
    display_ner_results(entities, entity_categories, entity_counts, chunked_tree)
    
    # Step 12: Save results
    output_file = "ner_analysis_results.csv"
    print(f"\\nStep 8: Saving results to {output_file}...")
    success = save_ner_results(ner_df, entities, entity_categories, output_file)
    
    if success:
        print("\\nNER analysis completed successfully!")
        print(f"Results saved to: {output_file}")
        print("Summary saved to: ner_analysis_results_summary.txt")
    else:
        print("\\nNER analysis completed with some errors in saving.")
    
    # Display sample of the DataFrame
    print("\\nSample of NER DataFrame:")
    print(ner_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()