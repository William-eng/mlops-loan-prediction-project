"""
Unit tests for model predictions
"""

import pytest
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions
import mlflow

# Set MLflow URI
mlflow.set_tracking_uri(config.TRACKING_URI)

@pytest.fixture
def single_prediction():
    """
    Fixture: Runs once and provides data to all tests
    Like a helper that prepares test data
    """
    # Load test data
    test_dataset = load_dataset(config.TEST_FILE)
    
    # Take first row
    single_row = test_dataset[config.FEATURES][:1]
    
    # Make prediction
    result = generate_predictions(single_row.to_dict('records'))
    
    return result


def test_single_pred_not_none(single_prediction):
    """
    Test 1: Prediction should not be None
    """
    assert single_prediction is not None, "Prediction returned None!"
    print("✓ Test passed: Prediction is not None")


def test_single_pred_str_type(single_prediction):
    """
    Test 2: Prediction should be a string
    """
    prediction_value = single_prediction.get('prediction')[0]
    assert isinstance(prediction_value, str), f"Expected string, got {type(prediction_value)}"
    print("✓ Test passed: Prediction is string type")


def test_single_pred_validate(single_prediction):
    """
    Test 3: Prediction should be either 'Y' or 'N'
    """
    prediction_value = single_prediction.get('prediction')[0]
    assert prediction_value in ['Y', 'N'], f"Invalid prediction: {prediction_value}"
    print("✓ Test passed: Prediction is valid ('Y' or 'N')")


def test_model_accuracy():
    """
    Test 4: Model accuracy on test set should be > 75%
    """
    from sklearn.metrics import accuracy_score
    
    # Load test data
    test_data = load_dataset(config.TEST_FILE)
    X_test = test_data[config.FEATURES]
    y_test = test_data[config.TARGET].map({'N': 0, 'Y': 1})
    
    # Get predictions
    predictions = generate_predictions(X_test.to_dict('records'))
    y_pred = [1 if p == 'Y' else 0 for p in predictions['prediction']]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy on test set: {accuracy:.4f}")
    assert accuracy > 0.75, f"Accuracy {accuracy:.4f} is below 75% threshold"
    print("✓ Test passed: Accuracy is above 75%")


if __name__ == "__main__":
    # Run with: pytest tests/test_prediction.py -v
    pytest.main([__file__, '-v'])