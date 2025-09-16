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
print("lemmetization",lemmatizer.lemmatize('running',pos='v'))
print("Stemming",ps.stem('running'))
print(lemmatizer.lemmatize('better',pos='a'))
print(ps.stem('better'))


text = "The striped bats are hanging on their feet for best"

for i in text.split():
    print(f"{text} | Stemming: {ps.stem(i)} | Lemmetization: {lemmatizer.lemmatize(i, pos='v')}")

word_pos = {'running':'v','better':'a','cats':'n','fairly':'r','quickly':'r','worst':'a','feet':'n'}
for key in word_pos.keys():
    print(f"{key} | Stemming: {ps.stem(key)} | Lemmetization: {lemmatizer.lemmatize(key, pos=word_pos[key])}")
    for word, pos in word_pos.items():
        lemma = lemmatizer.lemmatize(word, pos=(pos))
        print(f"{word} (POS: {pos}): Lemmatized: {lemma}")



#different 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
nltk.download('stopwords')
text = "The striped bats are hanging on their feet for best"
print("original text",text)
#tokenize and pos tag
text = re.sub('[^a-zA-Z]',' ',text)
text = text.lower().split()
print("tokenized text",text)

cleaned_text = [word for word in text if word not in STOPWORDS]
print("cleaned text",cleaned_text)

pos_tags = nltk.pos_tag(cleaned_text)
print("pos tags",pos_tags)

file = open('C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/NLP/nltp/preprocessing/testing.txt','w')
for tag in pos_tags:
    file.write(str(tag)+'\n')
file.close()