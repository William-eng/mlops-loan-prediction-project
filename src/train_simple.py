"""
Simple training script - no fancy stuff yet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import preprocessing as pp
import joblib

print("=" * 50)
print("STEP 1: Load Data")
print("=" * 50)

# Load data
train_data = pd.read_csv('../data/raw/train.csv')
print(f"Loaded {len(train_data)} training samples")

# Separate features and target
X = train_data.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = train_data['Loan_Status'].map({'N': 0, 'Y': 1})  # Convert to numbers

print(f"Features: {list(X.columns)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

print("\n" + "=" * 50)
print("STEP 2: Split Data")
print("=" * 50)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print("\n" + "=" * 50)
print("STEP 3: Build Pipeline")
print("=" * 50)

# Define which columns are numeric and categorical
NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
CAT_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 
                'Self_Employed', 'Credit_History', 'Property_Area']

# Build preprocessing + model pipeline
pipeline = Pipeline([
    ('DomainProcessing', pp.DomainProcessing(
        variable_to_modify=['ApplicantIncome'],
        variable_to_add='CoapplicantIncome'
    )),
    ('MeanImputation', pp.MeanImputer(variables=NUM_FEATURES)),
    ('ModeImputation', pp.ModeImputer(variables=CAT_FEATURES)),
    ('DropFeatures', pp.DropColumns(variables_to_drop=['CoapplicantIncome'])),
    ('LabelEncoder', pp.CustomLabelEncoder(variables=CAT_FEATURES)),
    ('LogTransform', pp.LogTransforms(variables=['ApplicantIncome', 'LoanAmount'])),
    ('MinMaxScale', MinMaxScaler()),
    ('RandomForest', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])

print("Pipeline created with steps:")
for name, _ in pipeline.steps:
    print(f"  - {name}")

print("\n" + "=" * 50)
print("STEP 4: Train Model")
print("=" * 50)

# Train
print("Training... (this may take a minute)")
pipeline.fit(X_train, y_train)
print("✓ Training complete!")

print("\n" + "=" * 50)
print("STEP 5: Evaluate Model")
print("=" * 50)

# Predict on validation set
y_pred = pipeline.predict(X_val)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 Score: {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred, 
                          target_names=['Rejected (N)', 'Approved (Y)']))

print("\n" + "=" * 50)
print("STEP 6: Save Model")
print("=" * 50)

# Create models directory
import os
os.makedirs('../models', exist_ok=True)

# Save the entire pipeline
model_path = '../models/loan_model.pkl'
joblib.dump(pipeline, model_path)
print(f"✓ Model saved to: {model_path}")

print("\n" + "=" * 50)
print("STEP 7: Test Prediction")
print("=" * 50)

# Test with one sample
sample = X_val.iloc[0:1]
prediction = pipeline.predict(sample)
prediction_proba = pipeline.predict_proba(sample)

print("Sample input:")
print(sample.to_dict('records')[0])
print(f"\nPrediction: {'Approved (Y)' if prediction[0] == 1 else 'Rejected (N)'}")
print(f"Confidence: {prediction_proba[0][prediction[0]]:.2f}")

print("\n" + "=" * 50)
print("✓ TRAINING COMPLETE!")
print("=" * 50)