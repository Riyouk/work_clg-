# 1. Import Required Libraries
import pandas as pd            # For data handling and manipulation
import numpy as np             # For numerical operations
import seaborn as sns          # For visualization
import matplotlib.pyplot as plt# For plotting graphs

from sklearn.preprocessing import StandardScaler  # For scaling features
from sklearn.model_selection import train_test_split  # For splitting data
from scipy.stats.mstats import winsorize          # For handling outliers
from sklearn.linear_model import LogisticRegression # The core model
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, RocCurveDisplay)              # For evaluating model

# 2. Load the Dataset
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/breast-cancer.csv")

# 3. Preview the Data
print("First few rows of the data:")
print(df.head())

# 4. Explore Target Variable
print("Counts for each diagnosis type:")
print(df["diagnosis"].value_counts())

# 5. Check Data Info
df["diagnosis"] = df["diagnosis"].astype("category")  # Make sure diagnosis is a category
print("Info about DataFrame:")
print(df.info())

# 6. Show Column Names
print("Columns in the dataset:")
print(df.columns)

# 7. Set Target (y) and Features (x)
y = "diagnosis"
x = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
    'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# 8. Drop Irrelevant Columns (like an ID)
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

print("Check DataFrame info after dropping ID:")
print(df.info())

# 9. Visualize Data Relationships
print("Pairplot by diagnosis (this might take time with larger datasets)...")
sns.pairplot(df, hue="diagnosis")  # Visualize feature distributions and relationships
plt.show()

# 10. Check Correlations Between Features
corr = df.corr(numeric_only=True)
print("Correlation matrix:")
print(corr)
# Uncomment for heatmap:
# plt.figure(figsize=(16,12))
# sns.heatmap(corr, cmap="crest", annot=True) 
# plt.show()

# 11. Winsorize (Cap Outliers)
def winsorize_data(data, limits=(0.02, 0.02)):
    """Limit extreme values in float columns to reduce the influence of outliers."""
    for col in data.select_dtypes(include=["float64"]):
        data.loc[:, col] = winsorize(data[col], limits=limits)
    return data

df_win = winsorize_data(df.copy())

# 12. Split Data into Training and Testing Sets
x_train, x_test = train_test_split(df_win, test_size=0.3, random_state=42)
# NOTE: x_train and x_test are full DataFrames for now

# 13. Feature Scaling (Standardization)
scaler = StandardScaler()
x_train[x] = scaler.fit_transform(x_train[x])
x_test[x] = scaler.transform(x_test[x])

print("First 5 rows of scaled training data:")
print(x_train.head())

# 14. Model Training: Logistic Regression
model = LogisticRegression(class_weight="balanced")  # Helps with class imbalance
model.fit(x_train[x], x_train[y])

# 15. Predict on Test Set
y_pred = model.predict(x_test[x])

# 16. Combine Predictions and Actuals
result = pd.DataFrame()
result["y"] = x_test[y].values
result["y_pred"] = y_pred

print("Predictions vs. Actual Labels:")
print(result.head())
print("Prediction probabilities for each class (first 5 rows):")
print(model.predict_proba(x_test[x])[:5])

# 17. Model Evaluation
print("Confusion Matrix:")
print(confusion_matrix(result['y'], result['y_pred']))
print("\nClassification Report:")
print(classification_report(result['y'], result['y_pred']))

# 18. Visualize Confusion Matrix 
con_mat = confusion_matrix(result['y'], result['y_pred'])
sns.heatmap(con_mat, annot=True, cmap="crest")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
