import os
from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


API_KEY = os.environ.get("API_KEY", "jakis_klucz")

X_train = np.array([[1], [2], [3], [4], [5], [6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)

@app.route("/")
def home():
    return jsonify({"message": "Witaj!", "api_key": API_KEY})

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Dane wejściowe muszą być w formacie JSON"}), 400
    data = request.get_json()
    if "X" not in data:
        return jsonify({"error": "Brak danych X"}), 400

    X_input = np.array(data["X"]).reshape(-1, 1)
    prediction = model.predict(X_input).tolist()
    return jsonify({"prediction": prediction, "api_key_used": API_KEY})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # 5000 lokalnie
    app.run(host="127.0.0.1", port=port)
