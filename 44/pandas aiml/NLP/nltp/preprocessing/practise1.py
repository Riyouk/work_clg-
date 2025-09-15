import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import re




sentence = "Lorem ipsum dolor, sit amet consectetur adipisicing elit. Minus dicta nemo, labore consequatur dolorum odit maxime sequi neque at nesciunt sunt repellendus perspiciatis cupiditate, totam sed provident et explicabo quaerat corrupti laudantium fugit dolores vitae! Eum tempora iste magnam excepturi? Animi reprehenderit libero aut molestias adipisci voluptas atque asperiores laboriosam debitis deserunt aspernatur beatae est itaque, hic numquam fugiat fugit cupiditate, veniam nam error quae voluptatibus facilis deleniti eius. Iusto laboriosam harum inventore laborum, id odit deserunt excepturi."

#retain only words
text = re.sub('[^a-zA-Z]',' ',sentence)
print(text)

# transform one form
text = text.lower()
print("\n",text)

#tokenize 
words = text.split()
print("\n",words)

print("remove stopwords")
ps = PorterStemmer()
text = [ps.stem(word) for word in words if word not in STOPWORDS]

print("\n the base words",text)
# def clean_text(text):


from nltk import ngrams 

#generate n-grams 
for i in text:
    ugram = list(ngrams(i,1))
    print("\n ugram",ugram)

    bigram = list(ngrams(i,2))
    print("\n bigram",bigram)

    trigram = list(ngrams(i,3))
    print("\n trigram",trigram)