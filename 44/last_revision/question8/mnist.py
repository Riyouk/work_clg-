import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28), name="Input_Layer"),
    Dense(256, activation='relu', name="HL-1"),
    Dropout(0.2),
    Dense(128, activation='relu', name="HL-2"),
    Dropout(0.2),
    Dense(64, activation='relu', name="HL-3"),
    Dropout(0.2),
    Dense(10, activation='softmax', name="Output_Layer")
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath='best_mnist_model.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    x_train, y_train,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
    callbacks=callbacks
)

print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

print("\nGenerating classification report...")
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

report = classification_report(y_test, predicted_labels, digits=4)
print("\nClassification Report:\n", report)
