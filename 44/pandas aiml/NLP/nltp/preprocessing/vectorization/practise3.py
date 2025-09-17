import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# oper = input("Enter text to preprocess: ")

# 
import contractions
# t = "I'm fine"
# t_ = contractions.fix(t)
# print("Before:", t)
# print("After:", t_)


#load data 
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/IMDB Dataset.csv",nrows=5000)
print(df.head())

print(df.info())

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

print(df.duplicated().sum())
# df = df.drop_duplicates()   
print(df.shape)

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def preprocess_reviews(text):
    if not isinstance(text, str):
        return ""
    text = contractions.fix(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    # text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

df["cleaned_reviews"] = df["review"].apply(preprocess_reviews)
print(df[["review","cleaned_reviews"]].head(10))

# cv = CountVectorizer(max_features=1500)
tf = TfidfVectorizer(max_features=1500)
X = tf.fit_transform(df["cleaned_reviews"]).toarray()
# print(X.shape)
print(X[0])
#label encoding 
le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])
print(df["sentiment"].value_counts())

y = df["sentiment"].values
# print(y.shape)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(np.bincount(y_train), np.bincount(y_test))

#scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)   
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train) 
# X_test = scaler.transform(X_test)

#model
model = RandomForestClassifier(random_state=42, n_jobs=-1,n_estimators=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluation
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# #postive reviews
# pos_text = " ".join(df[df["feedback"]==1]["cleaned_reviews"].values)
# # print(pos_text)
# wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens', max_words=200).generate(pos_text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wc_pos, interpolation='bilinear')
# plt.axis('off')
# plt.title("Positive Reviews Word Cloud")
# plt.show()

# #negative reviews
# neg_text = " ".join(df[df["feedback"]==0]["cleaned_reviews"].values)
# # print(neg_text)
# wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds', max_words=200).generate(neg_text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wc_neg, interpolation='bilinear')
# plt.axis('off')
# plt.title("Negative Reviews Word Cloud")
# plt.show()

# #feature importance
# # importances = model.feature_importances_
# # indices = np.argsort(importances)[-10:]  # Top 10 features
# # features = np.array(cv.get_feature_names_out())[indices]
# # plt.figure(figsize=(10, 5))
# # plt.title("Feature Importances")
# # plt.barh(range(len(indices)), importances[indices], align='center') 
# # plt.yticks(range(len(indices)), features)
# # plt.xlabel("Relative Importance")
# # plt.show()

# # example 
# new_review = ["Alexa is amazing,I use it ever day and it works prefectly!"]

# #clean it using the same function 
# new_review_cleaned = [preprocess_reviews(review) for review in new_review]
# # new_review_cleaned = [preprocess_reviews(new_review[0])]

# #transform it using the same CountVectorizer
# x_new = cv.transform(new_review_cleaned).toarray()
# print(x_new)

# # pridict sentiment
# prediction = model.predict(x_new)

# print(f"Review: {new_review[0]}")
# print(f"Predicted Sentiment: {'Positive 😁' if prediction[0]==1 else 'Negative 😒'}")

# #example 2
# new_review2 = ["I hate alexa, it is very bad and not useful at all",
#                 "I love alexa, it is very good and useful at all",
#                 "Alexa is okay, it works sometimes but not always",
#                 "Alexa is terrible, it never works and is a waste of money",
#                 "Alexa is fantastic, it works perfectly and is worth every penny"]

# for review in new_review2:
#     new_review_cleaned2 = [preprocess_reviews(review)]
#     x_new2 = cv.transform(new_review_cleaned2).toarray()
#     prediction2 = model.predict(x_new2)
#     print(f"Review: {review}")
#     print(f"Predicted Sentiment: {'Positive 😁' if prediction2[0]==1 else 'Negative 😒'}")
#     print("-"*50)

# #DESICION TREE 
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train, y_train)  

# y_pred_dt = dt_model.predict(X_test)
# acc_dt = accuracy_score(y_test, y_pred_dt)
# print(f"Decision Tree Accuracy: {acc_dt*100:.2f}%")
# cm_dt = confusion_matrix(y_test, y_pred_dt)
# disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=dt_model.classes_)
# disp_dt.plot(cmap=plt.cm.Blues)
# plt.show()

