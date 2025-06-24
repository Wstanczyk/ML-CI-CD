from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_predict():
    # Wczytaj dane
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Podziel na train i test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Trenuj model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predykcje
    preds = model.predict(X_test)

    return preds.tolist(), X_test.tolist()

def get_accuracy():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
