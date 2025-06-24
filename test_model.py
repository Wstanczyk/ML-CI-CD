from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    preds, X_test = train_and_predict()
    assert len(preds) > 0, "Predictions should not be empty."
    assert len(preds) == len(X_test), "Number of predictions should match number of test samples."

def test_predictions_value_range():
    preds, _ = train_and_predict()
    assert all(p in [0,1,2] for p in preds), "Predictions values should be in [0,1,2]."

def test_model_accuracy():
    accuracy = get_accuracy()
    assert accuracy >= 0.7, f"Model accuracy should be at least 70%, got {accuracy*100:.2f}%."
