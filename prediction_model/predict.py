"""
Load best model from MLflow and make predictions
"""

import pandas as pd
import numpy as np
import mlflow
from prediction_model.config import config

# Set MLflow URI
mlflow.set_tracking_uri(config.TRACKING_URI)

def generate_predictions(data_input):
    """
    Make predictions using the best model from MLflow
    
    Args:
        data_input: List of dictionaries, each containing loan application data
        
    Example:
        data = [{
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': '0',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': 5000,
            'CoapplicantIncome': 2000,
            'LoanAmount': 150,
            'Loan_Amount_Term': 360,
            'Credit_History': 1.0,
            'Property_Area': 'Urban'
        }]
        result = generate_predictions(data)
        # Result: {'prediction': ['Y']}
    
    Returns:
        Dictionary with 'prediction' key containing list of predictions
    """
    # Convert to DataFrame
    data = pd.DataFrame(data_input)
    
    print("🔍 Finding best model from MLflow...")
    
    # Get experiment
    experiment_name = config.EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow!")
    
    experiment_id = experiment.experiment_id
    
    # Search runs, order by F1 score (highest first)
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=['metrics.f1_score DESC']
    )
    
    if len(runs_df) == 0:
        raise ValueError("No runs found in experiment!")
    
    # Get best run
    best_run = runs_df.iloc[0]
    best_run_id = best_run['run_id']
    best_f1 = best_run['metrics.f1_score']
    
    print(f"✓ Best model found!")
    print(f"  Run ID: {best_run_id}")
    print(f"  F1 Score: {best_f1:.4f}")
    
    # Load model
    best_model_uri = f"runs:/{best_run_id}/{config.MODEL_ARTIFACT_PATH}"


    loan_prediction_model = mlflow.sklearn.load_model(best_model_uri)
    
    # Make predictions
    prediction = loan_prediction_model.predict(data)
    
    # Convert to Y/N
    output = np.where(prediction == 1, 'Y', 'N')
    
    result = {"prediction": output.tolist()}
    return result


def generate_predictions_batch(data_input):
    """
    Same as generate_predictions but for batch processing
    Takes a pandas DataFrame instead of list of dicts
    """
    print("🔍 Finding best model from MLflow...")
    
    experiment_name = config.EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    
    runs_df = mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=['metrics.f1_score DESC']
    )
    
    best_run = runs_df.iloc[0]
    best_run_id = best_run['run_id']
    
    print(f"✓ Using run: {best_run_id}")
    
    best_model_uri = f"runs:/{best_run_id}/{config.MODEL_ARTIFACT_PATH}"

    loan_prediction_model = mlflow.sklearn.load_model(best_model_uri)

    
    prediction = loan_prediction_model.predict(data_input)
    output = np.where(prediction == 1, 'Y', 'N')
    
    result = {"prediction": output}
    return result


if __name__ == '__main__':
    # Test prediction
    sample_data = [{
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }]
    
    result = generate_predictions(sample_data)
    print(f"\n✓ Prediction: {result['prediction'][0]}")