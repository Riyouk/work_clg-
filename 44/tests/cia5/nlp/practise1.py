
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

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/amazon_alexa.tsv",sep='\t')

print("Initial Data Sample:")
print(df.head())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"\nCleaned Data Shape: {df.shape}")


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = contractions.fix(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

# Apply preprocessing to reviews
df["cleaned_reviews"] = df["verified_reviews"].apply(preprocess_text)
print("\nSample of cleaned reviews:")
print(df[["verified_reviews", "cleaned_reviews"]].head(3))


cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(df["cleaned_reviews"]).toarray()

y = df["feedback"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain/Test shapes: {X_train.shape}, {X_test.shape}")


model = RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

print("\n classification report : ",classification_report(y_test,y_pred))
