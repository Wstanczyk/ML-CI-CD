from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

X_train = np.array([[1], [2], [3], [4], [5], [6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)

def validate_input(data):
    if not data:
        return "Brak danych", 400
    return None, 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Witaj!"})

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Dane wejściowe muszą być w formacie JSON"}), 400

    data = request.get_json()
    error_message, status_code = validate_input(data)
    if error_message:
        return jsonify({"error": error_message}), status_code

    X_input = np.array(data["X"]).reshape(-1, 1)
    prediction = model.predict(X_input).tolist()
    return jsonify({"prediction": prediction})

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model_type": "LogisticRegression",
        "num_features": model.coef_.shape[1],
        "num_classes": len(np.unique(y_train))
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
