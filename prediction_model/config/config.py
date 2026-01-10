"""
Configuration file - All settings in ONE place
"""

import os

# Get the directory where this file is located
current_directory = os.path.dirname(os.path.realpath(__file__))
PACKAGE_ROOT = os.path.dirname(current_directory)

# Data paths
DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# Model configuration
TARGET = 'Loan_Status'

# Features used in the model
FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Numerical features (need mean imputation)
NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Categorical features (need mode imputation)
CAT_FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'Credit_History', 'Property_Area'
]

# Features to encode (convert text to numbers)
FEATURES_TO_ENCODE = CAT_FEATURES.copy()

# Domain-specific transformations
FEATURE_TO_MODIFY = ['ApplicantIncome']  # Will add CoapplicantIncome to this
FEATURE_TO_ADD = 'CoapplicantIncome'
DROP_FEATURES = ['CoapplicantIncome']  # Drop after adding

# Log transformations (reduce skewness)
LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']

# MLflow settings (we'll set these up later)
TRACKING_URI = "http://localhost:5000"  # Local for now
EXPERIMENT_NAME = "loan_prediction_model"
MODEL_NAME = "/Loanprediction-model"

# AWS settings (we'll configure these when ready)
S3_BUCKET = "loanprediction"
FOLDER = "datadrift"

print("✓ Configuration loaded successfully")