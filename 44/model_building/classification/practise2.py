# ===============================
# Step 1: Import Dependencies
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# Step 2: Load Dataset
# ===============================
data = pd.read_csv("iris.csv")  # Save your dataset as iris.csv

print("First 5 rows of dataset:")
print(data.head())
print("\nData Info:")
print(data.info())

# ===============================
# Step 3: Data Cleaning
# ===============================

# 3.1 Drop Id column (not useful)
if "Id" in data.columns:
    data.drop("Id", axis=1, inplace=True)

# 3.2 Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# 3.3 Check duplicates
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    data.drop_duplicates(inplace=True)

# ===============================
# Step 4: Outlier Detection
# ===============================
# Using IQR method for numeric columns
def detect_outliers(df, col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col_name] < lower) | (df[col_name] > upper)]
    return outliers

for col in data.columns[:-1]:  # Skip Species column
    outliers = detect_outliers(data, col)
    print(f"{col} has {len(outliers)} outliers")

# (Note: In real datasets, we might drop or transform them.
# For Iris, we usually keep them as they represent natural variation!)

# ===============================
# Step 5: Encode Target Labels
# ===============================
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])
print("\nEncoded Species Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

# ===============================
# Step 6: Train-Test Split
# ===============================
X = data.drop("Species", axis=1)
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape}")

# ===============================
# Step 7: Feature Standardization
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Step 8: Build Logistic Regression Model
# ===============================
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
model.fit(X_train_scaled, y_train)

# ===============================
# Step 9: Model Evaluation
# ===============================
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# Step 10: Testing with New Sample
# ===============================
sample = np.array([[5.0, 3.3, 1.4, 0.2]])  # Example sample
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("\nPredicted Species for sample:", le.inverse_transform(prediction))