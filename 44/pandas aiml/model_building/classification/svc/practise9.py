# Import necessary libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, RocCurveDisplay
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/breast-cancer.csv")

# Preview first rows
print("Dataset sample:")
print(df.head())

# Check class distribution in target variable
print("\nDiagnosis value counts:")
print(df["diagnosis"].value_counts())

# Convert diagnosis column to category type for clarity
df["diagnosis"] = df["diagnosis"].astype("category")
print("\nDataFrame info with diagnosis as category:")
print(df.info())

# List the columns/features in dataset
print("\nColumns in dataset:")
print(df.columns)

# Define target and features list
y = ["diagnosis"]
x = [   # Selected features relevant for model
    'radius_mean', 'texture_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# Drop 'id' column because it's non-informative for prediction
df.drop(columns=["id"], inplace=True)
print("\nDataFrame info after dropping 'id':")
print(df.info())

# Compute and print correlation matrix among numeric features
corr = df.corr(numeric_only=True)
print("\nCorrelation matrix sample:")
print(corr)

# Encode target labels ('M' -> 1, 'B' -> 0) for classification
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

# Define a function to winsorize numeric columns to limit extreme outliers
def winsorize_data(data, limits=(0.02, 0.02)):
    for col in data.select_dtypes(include=["float64"]):
        # Apply winsorization - clip extreme 2% low/high values 
        data.loc[:, col] = winsorize(data[col], limits=limits)
    return data

# Apply winsorization on dataframe copy
df_win = winsorize_data(df.copy())

# Split dataset into training and testing sets with stratification on diagnosis to keep class ratio
x_train, x_test = train_test_split(
    df_win, test_size=0.30, random_state=42, stratify=df_win['diagnosis']
)

# Feature scaling: standardize features (mean=0, std=1)
scaler = StandardScaler()
x_train[x] = scaler.fit_transform(x_train[x])
x_test[x] = scaler.transform(x_test[x])  # Use same scaler on test data

print("\nScaled training data preview:")
print(x_train[x].head())

print("\nScaled test data preview:")
print(x_test[x].head())

# Initialize and train SVM model with default parameters
model = SVC(probability=True, random_state=42)  # probability=True to enable ROC curve plotting
model.fit(x_train[x], x_train[y].values.ravel())  # .ravel() to convert to 1D array

# Predict on test set
y_pred = model.predict(x_test[x])

# Prepare results dataframe for comparison
result = pd.DataFrame({
    "Actual": x_test[y].values.ravel(),
    "Predicted": y_pred
})

print("\nPredictions vs Actual:")
print(result.head())

# Evaluate model with confusion matrix
conf_mat = confusion_matrix(result["Actual"], result["Predicted"])
print("\nConfusion Matrix:")
print(conf_mat)

# Visualize confusion matrix using heatmap
sns.heatmap(conf_mat, annot=True, cmap="crest", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report gives precision, recall, f1-score
print("\nClassification Report:")
print(classification_report(result["Actual"], result["Predicted"]))

# Accuracy score
acc = accuracy_score(result["Actual"], result["Predicted"])
print("\nAccuracy Score:", acc)

# Plot ROC Curve using model's probabilities and test labels
RocCurveDisplay.from_estimator(model, x_test[x], x_test[y].values.ravel())
plt.title("ROC Curve from estimator")
plt.show()

# Plot ROC Curve using true labels and predicted values
RocCurveDisplay.from_predictions(result["Actual"], result["Predicted"])
plt.title("ROC Curve from predictions")
plt.show()
