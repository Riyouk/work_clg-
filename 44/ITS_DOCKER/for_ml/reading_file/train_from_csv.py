import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load CSV
data = pd.read_csv("breast-cancer.csv")

# Step 2: Preprocess target
# Convert 'M' -> 0 (malignant), 'B' -> 1 (benign)
data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1})

# Step 3: Separate features and target
# Drop 'id' and 'diagnosis' from features
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Step 4: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 7: Evaluate
preds = model.predict(X_test_scaled)
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=['malignant','benign']))

# Step 8: Save model and scaler
joblib.dump(model, "breast_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved model and scaler as 'breast_model.pkl' and 'scaler.pkl'")
