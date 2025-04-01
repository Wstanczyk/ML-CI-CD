from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

X_train = np.array([[1], [2], [3], [4], [5], [6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Witaj!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "X" not in data:
        return jsonify({"error": "Brak danych"}), 400

    X_input = np.array(data["X"]).reshape(-1, 1)
    prediction = model.predict(X_input).tolist()

    return jsonify({"predykcja": prediction})

if __name__ == "__main__":
    app.run(debug=True)
