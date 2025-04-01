from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

#wczytanie danych
dataset = load_iris()
X = dataset.data
y = dataset.target

#treningowy i testowy (7/3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = keras.Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

#kompilacja
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#trenowanie
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))



loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

joblib.dump(model, "model_v1.joblib")

