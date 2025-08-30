import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,RocCurveDisplay
# from sklearn.tree import DecisionTreeClassifier,export_text,plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/breast-cancer.csv")


print(df.head())
# print(df.info())

# print(df.describe())

print(df["diagnosis"].value_counts())

df["diagnosis"] = df["diagnosis"].astype("category")
print(df.info())

print(df.columns)

y = ["diagnosis"]
x = ['radius_mean', 'texture_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

df.drop(columns=["id"],inplace=True)
# df.drop(['perimeter_mean', 'area_mean','perimeter_worst', 'area_worst', 'perimeter_se', 'area_se', 'compactness_mean','concave points_mean', 'compactness_se', 'concave points_se','compactness_worst','concave points_worst'], axis=1, inplace=True)
print(df.info())

# sns.pairplot(df,hue="diagnosis")
# plt.figure(figsize=(20,15))
# plt.show()

# sns.boxplot(df)
# plt.show()

# numreic_col = df.select_dtypes(include=["float64"])
# corr = numreic_col.corr()
corr = df.corr(numeric_only=True)
# plt.figure(figsize=(16,12))
# sns.heatmap(corr,cmap="crest",annot=True) 
# plt.show()

print(corr)

def winsorize_data(data,limits=(0.02,0.02)):
    for col in data.select_dtypes(include=["float64"]):
        data.loc[:,col] = winsorize(data[col], limits=limits)
    return data

df_win = winsorize_data(df)
X = df_win[x]
y = df_win["diagnosis"]

# Define features and target
X = df_win[x]   # features
y = df_win["diagnosis"]   # target (Series, not list)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
print("\n--- Starting Hyperparameter Tuning ---")
start_time = time.time()

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 4, 8, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Create base model
rf = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and best score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score: {:.4f}".format(grid_search.best_score_))
print("Time taken for GridSearchCV: {:.2f} seconds".format(time.time() - start_time))

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_predict = best_model.predict(X_test_scaled)

# Evaluation
print("\n--- Model Evaluation with Best Parameters ---")
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))

# Feature importance
feature_importance = pd.DataFrame(
    {'feature': x,
     'importance': best_model.feature_importances_}
).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))  # Show top 10 features

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Random Forest Feature Importance (Top 10)')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show()
