
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


df = pd.read_csv("C:/Users/hp/Desktop/GHOST/work_clg-/44/pandas aiml/DataSets/DataSets/IMDB Dataset.csv")

print("Initial Data Sample:")
print(df.head())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"/nCleaned Data Shape: {df.shape}")


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = contractions.fix(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

# Apply preprocessing to reviews
df["cleaned_reviews"] = df["review"].apply(preprocess_text)
print("/nSample of cleaned reviews:")
print(df[["verified_reviews", "cleaned_reviews"]].head(3))

# ========================
# WORD CLOUD GENERATION
# ========================

# Combine all cleaned reviews into one large string
# all_text = " ".join(df["cleaned_reviews"])

# # Generate the word cloud
# wordcloud = WordCloud(
#     width=1200,
#     height=600,
#     background_color='white',
#     max_words=200
# ).generate(all_text)

# # Display the word cloud
# plt.figure(figsize=(12, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud of Amazon Alexa Reviews", fontsize=18)
# plt.show()



cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(df["cleaned_reviews"]).toarray()

y = df["sentiment"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"/nTrain/Test shapes: {X_train.shape}, {X_test.shape}")


model = RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

print("/n classification report : ",classification_report(y_test,y_pred))
