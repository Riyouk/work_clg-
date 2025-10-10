import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report


print(" Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training data shape: {x_train.shape}, {y_train.shape}")
print(f"Testing data shape: {x_test.shape}, {y_test.shape}")


x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28), name="Input_Layer"),
    Dense(256, activation='relu', name="Hidden_Layer-1"),
    Dropout(0.2),
    Dense(128, activation='relu', name="Hidden_Layer-2"),
    Dropout(0.3),
    Dense(64, activation='relu', name="Hidden_Layer-3"),
    Dropout(0.3),
    Dense(10, activation='softmax', name="Output_Layer")
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining model...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)


print("\n Evaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n Test Accuracy: {test_acc:.4f}")


print("\nGenerating classification report...")
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

report = classification_report(y_test, predicted_labels, digits=4)
print("\nClassification Report:\n", report)
