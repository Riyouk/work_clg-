import pandas as pd 
import numpy as np 
import contractions
import re 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from nltk import ngrams
nltk.download("punkt",quiet=True)
nltk.download("stopwords",quiet=True)
nltk.download("wordnet",quiet=True)
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/revision/nlp/IMDB Dataset.csv",nrows=6000)
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df["sentiment"])

#type conversion 
df["sentiment"] = df["sentiment"].astype("category")

#label encoding 
label = LabelEncoder()
df["sentiment"] = label.fit_transform(df["sentiment"])
print(df["sentiment"])

lemetize = WordNetLemmatizer()
def preprocess(text):
    text = contractions.fix(text)
    text = re.sub("[^a-zA-Z]"," ",text)
    text = text.lower().split()
    # text = [lemetize.lemmatize(word) for word in text if word not in set(stopwords.words("english"))]
    text = [lemetize.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return "".join(text)

df["cleaned_review"] = df["review"].apply(preprocess)
print(df["cleaned_review"])