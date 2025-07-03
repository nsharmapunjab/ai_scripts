import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from model import preprocess, load_model

# Load test data
X_test = np.load("test_data/X_test.npy")
y_test = np.load("test_data/y_test.npy")
group_labels = np.load("test_data/group_labels.npy")  # For fairness testing

model = load_model()

# --- 1. Unit Test: Preprocessing ---
def test_preprocessing_range():
    sample = np.array([[5, 2, 100]])
    processed = preprocess(sample)
    assert (processed >= 0).all() and (processed <= 1).all()

# --- 2. Functional Test: Known Input ---
def test_known_prediction():
    known_input = X_test[0].reshape(1, -1)
    expected_label = y_test[0]
    prediction = model.predict(known_input)[0]
    assert prediction == expected_label

# --- 3. Performance Test ---
def test_model_accuracy():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.90

# --- 4. Regression Test: Compare with baseline ---
def test_against_baseline():
    new_accuracy = accuracy_score(y_test, model.predict(X_test))
    baseline_accuracy = 0.91  # Replace with baseline
    assert new_accuracy >= baseline_accuracy - 0.01

# --- 5. Robustness Test ---
def test_noisy_input_resilience():
    original_pred = model.predict(X_test[10].reshape(1, -1))[0]
    noisy_input = X_test[10] + np.random.normal(0, 0.01, size=X_test[10].shape)
    noisy_pred = model.predict(noisy_input.reshape(1, -1))[0]
    assert original_pred == noisy_pred  # Allow margin if using probabilistic model

# --- 6. Fairness Test ---
def test_group_fairness():
    y_pred = model.predict(X_test)
    groups = np.unique(group_labels)
    accuracies = []

    for group in groups:
        mask = group_labels == group
        acc = accuracy_score(y_test[mask], y_pred[mask])
        accuracies.append(acc)

    max_diff = max(accuracies) - min(accuracies)
    assert max_diff < 0.05

#How to run
pip install pytest numpy scikit-learn
pytest test_model.py -v
