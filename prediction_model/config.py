"""
Configuration settings for the prediction model
"""
from pathlib import Path
import os

print("✓ Configuration loaded successfully")

# Directories - Use Path objects
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Data paths
DATAPATH = PACKAGE_ROOT / "datasets"  # Path object
SAVE_MODEL_PATH = PROJECT_ROOT / "trained_models"  # Path object

# Create directories if they don't exist
DATAPATH.mkdir(parents=True, exist_ok=True)
SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)

# MLflow configuration
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "loan_prediction_experiment"

# Data files
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# Target variable
TARGET = "Loan_Status"

# Features - Update these based on your actual CSV columns
FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Property_Area'
]

# Numerical features
NUM_FEATURES = [
    'ApplicantIncome', 
    'CoapplicantIncome', 
    'LoanAmount', 
    'Loan_Amount_Term'
]

# Categorical features
CAT_FEATURES = [
    'Gender', 
    'Married', 
    'Dependents', 
    'Education',
    'Self_Employed', 
    'Property_Area', 
    'Credit_History'
]

# Features to encode (categorical)
FEATURES_TO_ENCODE = CAT_FEATURES.copy()

# Features to apply log transform
LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']

# Features to drop (if any - empty for now)
DROP_FEATURES = []

# Feature engineering parameters
FEATURE_TO_MODIFY = None
FEATURE_TO_ADD = None

# Model file name
MODEL_NAME = "loan_prediction_model.pkl"

# Print configuration on load
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CONFIGURATION SETTINGS")
    print("=" * 70)
    print(f"Package Root: {PACKAGE_ROOT}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Path: {DATAPATH}")
    print(f"Model Save Path: {SAVE_MODEL_PATH}")
    print(f"Train File: {TRAIN_FILE}")
    print(f"Target: {TARGET}")
    print(f"Number of Features: {len(FEATURES)}")
    print(f"Numerical Features: {NUM_FEATURES}")
    print(f"Categorical Features: {CAT_FEATURES}")
    print("=" * 70)