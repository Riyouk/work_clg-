# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet

# # Download resources (only first time)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# lemmatizer = WordNetLemmatizer()

# print(lemmatizer.lemmatize("running", pos="v"))   # run
# print(lemmatizer.lemmatize("studies", pos="n"))   # study
# print(lemmatizer.lemmatize("better", pos="a"))    # good
# print(lemmatizer.lemmatize("fairly", pos="r"))    # fairly (no change)


import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("running studies better children")
for token in doc:
    print(token.text, "→", token.lemma_)
