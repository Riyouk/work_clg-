import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re 
import contractions
import nltk 
from nltk.corpus import stopwords 
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from nltk import ngrams

nltk.download("stopwords",quiet=True)
nltk.download("wordnet",quiet=True)
nltk.download("punkt",quiet=True)
nltk.download("punkt_tab",quiet=True)
nltk.download("averaged_perceptron_tagger_eng",quiet=True)
nltk.download("maxent_ne_chunker",quiet=True)
nltk.download("maxent_ne_chunker_tab",quiet=True)
nltk.download("words",quiet=True)

df = pd.read_csv("IMDB Dataset.csv",nrows=5000)
print("sample of the data : ",df.head(10))
print("info : ",df.info())
print("describe: ",df.describe())
print("null values : ",df.isna().sum())

lematizer = WordNetLemmatizer()
stopword = set(stopwords.words("english"))

def clean_text(text):
    text = contractions.fix(text)
    text = re.sub("^[a-zA-z]"," ",text)
    text = text.lower().split()
    text = [lematizer.lemmatize(word) for word in text if word not in stopword]
    return " ".join(text)

def gen_ngrams(words,n):
    words = nltk.word_tokenize(words)
    return list(ngrams(words,n))

def post_tagging(words):
    return nltk.pos_tag(words)

def ner_chunking(pos_tags):
    return nltk.ne_chunk(pos_tags)


df["cleaned_reviews"] = df["review"].apply(clean_text)
print("sample of cleaned reviews : ")
print([df["cleaned_reviews"].head(3),df["review"].head(3)])

# # N-grams
# df["unigrams"] = df["cleaned_reviews"].apply(lambda x: gen_ngrams(x, 1))
# df["bigrams"] = df["cleaned_reviews"].apply(lambda x: gen_ngrams(x, 2))
# df["trigrams"] = df["cleaned_reviews"].apply(lambda x: gen_ngrams(x, 3))

# # POS Tagging
# df["pos_tags"] = df["cleaned_reviews"].apply(lambda x: post_tagging(nltk.word_tokenize(x)))

# # NER
# df["ner"] = df["pos_tags"].apply(lambda x: ner_chunking(x))

# print("NER sample: ", df["ner"].head())


label = LabelEncoder()
df["sentiment"] = label.fit_transform(df["sentiment"])
# print(df["sentiment"].head())


# tfid = TfidfTransformer(max_features=1500)
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(df["cleaned_reviews"])
y = df["sentiment"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

model = RandomForestClassifier()
skfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
param_grid = {"max_depth" : [3,4],
              "max_features" : ["sqrt","log2"],
              "n_estimators" : [10,15,20]}

grid = GridSearchCV(estimator=model,cv=skfold,n_jobs=-1,param_grid=param_grid)
grid.fit(x_train,y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(x_test)

print("classification_report : ",classification_report(y_test,y_pred))
print("accuracy_score",accuracy_score(y_test,y_pred))


#wordcloud 
pos_wrds = " ".join(df[df["sentiment"]==1]["cleaned_reviews"].values)
wc_pos = WordCloud(width=800,height=400,background_color="white",colormap="Greens",max_words=200).generate(pos_wrds)
plt.figure(figsize=(10,5))
plt.imshow(wc_pos,interpolation="bilinear")
plt.axis("off")
plt.title("POSITIVE reviews word cloud")
plt.show()


neg_wrds = " ".join(df[df["sentiment"]==0]["cleaned_reviews"].values)
wc_neg = WordCloud(width=800,height=400,background_color="white",colormap="Reds",max_words=200).generate(neg_wrds)
plt.figure(figsize=(10,6))
plt.imshow(wc_neg,interpolation="bilinear")
plt.axis("off")
plt.title("NEGATIVE reviews word cloud")
plt.show()