import pandas as pd 
import numpy as np 

# 1
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/revision/cross validation/K_fold/imbalanced_fraud_dataset.csv")
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isna().sum())

#spliting data 
x = df.drop("is_fraud",axis=1)
y = df["is_fraud"]

#Transforming data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# 1.2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

log_reg = LogisticRegression()

scores = cross_val_score(log_reg, x, y, cv=kf, scoring="accuracy")

print("Fold accuracies:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())


# 1.3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

log_reg.fit(X_train, y_train)
pred = log_reg.predict(X_test)

print("Train-test accuracy:", accuracy_score(y_test, pred))


# 2
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import numpy as np

dt = DecisionTreeClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3
print("\n===== K-FOLD F1-SCORES =====")
kf_f1 = []

for train_idx, test_idx in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)

    f1 = f1_score(y_test, pred)
    kf_f1.append(f1)
    print("F1-score:", f1)

print("Mean F1 (KFold):", np.mean(kf_f1))


print("\n===== STRATIFIED K-FOLD F1-SCORES =====")
skf_f1 = []

for train_idx, test_idx in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)

    f1 = f1_score(y_test, pred)
    skf_f1.append(f1)
    print("F1-score:", f1)

print("Mean F1 (StratifiedKFold):", np.mean(skf_f1))



