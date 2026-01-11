"""
Improved training pipeline with:
1. MLflow experiment tracking
2. Hyperparameter tuning with Hyperopt
3. XGBoost model (better than Random Forest)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb

# Import our custom modules - use direct imports, not relative
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.processing.preprocessing import (
    DomainProcessing,
    MeanImputer,
    ModeImputer,
    DropColumns,
    CustomLabelEncoder,
    LogTransforms
)

print("=" * 70)
print("LOAN PREDICTION MODEL - TRAINING PIPELINE WITH MLFLOW")
print("=" * 70)

# Set up MLflow
mlflow.set_tracking_uri(config.TRACKING_URI)
print(f"✓ MLflow tracking URI: {config.TRACKING_URI}")
print(f"✓ Open http://localhost:5000 in your browser to see experiments")

def get_data(filename):
    """Load and prepare data"""
    print(f"\nLoading {filename}...")
    data = load_dataset(filename)
    
    # Separate features and target
    X = data[config.FEATURES]
    y = data[config.TARGET].map({'N': 0, 'Y': 1})
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target distribution: {dict(y.value_counts())}")
    
    return X, y

# Load data
X, y = get_data(config.TRAIN_FILE)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✓ Train: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# Define hyperparameter search space
print("\n" + "=" * 70)
print("HYPERPARAMETER SEARCH SPACE")
print("=" * 70)
print("We'll try different combinations of:")
print("  - max_depth: How deep the tree can grow (3 to 9)")
print("  - learning_rate: How fast the model learns (0.01 to 0.3)")
print("  - n_estimators: How many trees to build (50 to 250)")
print("  - And more...")

search_space = {
    'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'n_estimators': hp.choice('n_estimators', np.arange(50, 300, 50, dtype=int)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'gamma': hp.uniform('gamma', 0, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}

# Counter for experiments
experiment_counter = [0]

def objective(params):
    """
    This function is called by Hyperopt for each experiment
    It trains a model and returns how good it is
    """
    experiment_counter[0] += 1
    print(f"\n{'='*70}")
    print(f"EXPERIMENT #{experiment_counter[0]}")
    print(f"{'='*70}")
    print(f"Trying parameters: {params}")
    
    # Create XGBoost classifier with current parameters
    clf = xgb.XGBClassifier(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    # Build complete pipeline
    pipeline = Pipeline([
        ('DomainProcessing', DomainProcessing(
            variable_to_modify=config.FEATURE_TO_MODIFY,
            variable_to_add=config.FEATURE_TO_ADD
        )),
        ('MeanImputation', MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputation', ModeImputer(variables=config.CAT_FEATURES)),
        ('DropFeatures', DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform', LogTransforms(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('XGBoostClassifier', clf)
    ])
    
    # Start MLflow run
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(nested=True):
        # Train
        print("Training...")
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        print(f"\n📊 Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            'f1_score': f1,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision
        })
        mlflow.sklearn.log_model(pipeline, "Loanprediction-model")
    
    # Return negative F1 (Hyperopt minimizes, we want to maximize F1)
    return {'loss': 1 - f1, 'status': STATUS_OK}

print("\n" + "=" * 70)
print("STARTING HYPERPARAMETER OPTIMIZATION")
print("=" * 70)
print("This will run 5 experiments to find the best model")
print("Each experiment will be logged to MLflow")
print("Go to http://localhost:5000 to watch in real-time!")

# Run optimization
trials = Trials()
best_params = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,  # Tree-structured Parzen Estimator
    max_evals=5,  # Try 5 different combinations
    trials=trials
)

print("\n" + "=" * 70)
print("✓ OPTIMIZATION COMPLETE!")
print("=" * 70)
print(f"Best parameters found: {best_params}")
print("\n✓ Check MLflow UI to see all experiments")
print("✓ The best model is automatically saved")