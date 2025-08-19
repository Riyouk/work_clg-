# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import machine learning related libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set random seed for reproducibility
np.random.seed(42)

# ----- DATA LOADING -----
# Load the Iris dataset
print("\n----- LOADING DATASET -----")
df1 = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/Iris.csv", index_col=0)
df = df1.copy()  # Create a copy to preserve original data

# ----- DATA EXPLORATION -----
print("\n----- DATA EXPLORATION -----")
# Display first 10 rows of the dataset
print("\nFirst 10 rows of the dataset:")
print(df.head(10))

# Display the shape of the dataset (rows, columns)
print("\nDataset shape (rows, columns):")
print(df.shape)

# Display information about the dataset
print("\nDataset information:")
print(df.info())

# Display statistical summary of the dataset
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values count:")
print(df.isna().sum())

# Display unique values in the target variable
print("\nUnique species in the dataset:")
print(df["Species"].unique())

# ----- DATA VISUALIZATION -----
print("\n----- DATA VISUALIZATION -----")

# Create a pairplot to visualize relationships between features
plt.figure(figsize=(10, 8))
sns.pairplot(df, hue="Species")
plt.suptitle("Pairplot of Iris Dataset Features by Species", y=1.02)
plt.savefig("iris_pairplot.png")
plt.close()
print("Pairplot saved as 'iris_pairplot.png'")

# Visualize the distribution of features with boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.iloc[:, 0:4])
plt.title("Boxplot of Iris Features")
plt.savefig("iris_boxplot.png")
plt.close()
print("Boxplot saved as 'iris_boxplot.png'")

# ----- DATA PREPROCESSING -----
print("\n----- DATA PREPROCESSING -----")

# Convert Species to categorical type
df["Species"] = df["Species"].astype("category")
print("Converted 'Species' to categorical type")

# Handle outliers in SepalWidthCm using winsorization
print("\nHandling outliers in SepalWidthCm using winsorization")
print("Before winsorization:")
print(df["SepalWidthCm"].describe())

df["SepalWidthCm"] = winsorize(df["SepalWidthCm"], limits=[0.1, 0.1])

print("After winsorization:")
print(df["SepalWidthCm"].describe())

# Visualize the effect of winsorization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x=df1["SepalWidthCm"])
plt.title("Before Winsorization")

plt.subplot(1, 2, 2)
sns.boxplot(x=df["SepalWidthCm"])
plt.title("After Winsorization")

plt.tight_layout()
plt.savefig("winsorization_effect.png")
plt.close()
print("Winsorization effect plot saved as 'winsorization_effect.png'")

# Encode the target variable
print("\nEncoding target variable 'Species'")
encoder = LabelEncoder()
df["Species_Encoded"] = encoder.fit_transform(df["Species"])
print("Original Species labels:", list(df["Species"].unique()))
print("Encoded Species labels:", list(df["Species_Encoded"].unique()))
print("Mapping:", dict(zip(df["Species"].unique(), df["Species_Encoded"].unique())))

# Calculate correlation matrix for numerical features
print("\nCalculating correlation matrix for features")
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
corr = df[feature_cols].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="crest", annot=True, fmt=".2f")
plt.title("Correlation Matrix of Iris Features")
plt.savefig("correlation_matrix.png")
plt.close()
print("Correlation matrix saved as 'correlation_matrix.png'")

# Feature scaling for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
print("\nFeatures scaled using StandardScaler")
print("Scaled features summary:")
print(X_scaled_df.describe().round(2))

# ----- CLASSIFICATION MODEL (LOGISTIC REGRESSION) -----
print("\n----- CLASSIFICATION MODEL: LOGISTIC REGRESSION -----")

# Prepare features (X) and target variable (y) for classification
X_class = df[feature_cols]
y_class = df["Species_Encoded"]

# Split the data into training and testing sets (70% train, 30% test)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

print(f"Training set size: {X_train_class.shape[0]} samples")
print(f"Testing set size: {X_test_class.shape[0]} samples")

# Initialize and train the Logistic Regression model
log_model = LogisticRegression(max_iter=200, random_state=42)
log_model.fit(X_train_class, y_train_class)
print("Logistic Regression model trained successfully")

# Cross-validation for logistic regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_model, X_class, y_class, cv=kf, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Make predictions on the test set
y_pred_class = log_model.predict(X_test_class)

# Calculate accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"\nTest set accuracy: {accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class, 
                          target_names=encoder.classes_))

# Generate confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=encoder.classes_, 
            yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'")

# ----- MODEL INTERPRETATION (LOGISTIC REGRESSION) -----
print("\n----- MODEL INTERPRETATION: LOGISTIC REGRESSION -----")

# Get feature importance (coefficients)
coef = log_model.coef_
print("Feature coefficients for each class:")
for i, species in enumerate(encoder.classes_):
    print(f"\n{species}:")
    for j, feature in enumerate(feature_cols):
        print(f"{feature}: {coef[i][j]:.4f}")

# Visualize feature importance
plt.figure(figsize=(12, 8))
for i, species in enumerate(encoder.classes_):
    plt.subplot(3, 1, i+1)
    sns.barplot(x=feature_cols, y=coef[i])
    plt.title(f"Feature Importance for {species}")
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.savefig("feature_importance.png")
plt.close()
print("Feature importance plot saved as 'feature_importance.png'")

# ----- REGRESSION MODEL (LINEAR REGRESSION) -----
print("\n----- REGRESSION MODEL: LINEAR REGRESSION -----")

# For regression, we'll predict SepalLengthCm based on other features
print("\nPredicting SepalLengthCm based on other features")
X_reg = df[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_reg = df['SepalLengthCm']

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg)
print("Linear Regression model trained successfully")

# Cross-validation for linear regression
kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_reg = cross_val_score(linear_model, X_reg, y_reg, cv=kf_reg, scoring='r2')
print(f"\nCross-validation R² scores: {cv_scores_reg}")
print(f"Mean CV R²: {cv_scores_reg.mean():.4f} (±{cv_scores_reg.std():.4f})")

# Make predictions
y_pred_reg = linear_model.predict(X_test_reg)

# Evaluate regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\nRegression Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--')
plt.xlabel('Actual SepalLengthCm')
plt.ylabel('Predicted SepalLengthCm')
plt.title('Linear Regression: Actual vs Predicted')
plt.savefig("regression_results.png")
plt.close()
print("Regression results plot saved as 'regression_results.png'")

# Visualize regression coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x=['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], y=linear_model.coef_)
plt.title('Linear Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.savefig("regression_coefficients.png")
plt.close()
print("Regression coefficients plot saved as 'regression_coefficients.png'")

# ----- POLYNOMIAL REGRESSION -----
print("\n----- POLYNOMIAL REGRESSION -----")

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_reg)

# Get feature names for polynomial features
poly_feature_names = poly.get_feature_names_out(['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
print(f"Polynomial features created: {poly_feature_names}")

# Split polynomial data
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y_reg, test_size=0.3, random_state=42
)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_poly)
print("Polynomial Regression model trained successfully")

# Make predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate polynomial regression model
poly_mse = mean_squared_error(y_test_poly, y_pred_poly)
poly_rmse = np.sqrt(poly_mse)
poly_mae = mean_absolute_error(y_test_poly, y_pred_poly)
poly_r2 = r2_score(y_test_poly, y_pred_poly)

print("\nPolynomial Regression Model Evaluation:")
print(f"Mean Squared Error (MSE): {poly_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {poly_rmse:.4f}")
print(f"Mean Absolute Error (MAE): {poly_mae:.4f}")
print(f"R² Score: {poly_r2:.4f}")

# Compare linear vs polynomial regression
print("\nModel Comparison:")
print(f"Linear Regression R²: {r2:.4f}")
print(f"Polynomial Regression R²: {poly_r2:.4f}")
print(f"Improvement: {(poly_r2 - r2) * 100:.2f}%")

# Visualize polynomial regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test_poly, y_pred_poly, alpha=0.7)
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--')
plt.xlabel('Actual SepalLengthCm')
plt.ylabel('Predicted SepalLengthCm')
plt.title('Polynomial Regression: Actual vs Predicted')
plt.savefig("polynomial_regression_results.png")
plt.close()
print("Polynomial regression results plot saved as 'polynomial_regression_results.png'")

# ----- ANALYSIS SUMMARY -----
print("\n----- ANALYSIS SUMMARY -----")
print("\nClassification (Logistic Regression):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

print("\nRegression (Linear):")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

print("\nRegression (Polynomial):")
print(f"R² Score: {poly_r2:.4f}")
print(f"RMSE: {poly_rmse:.4f}")

print("\n----- ANALYSIS COMPLETE -----")