"""
Data preprocessing transformers for the ML pipeline
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainProcessing(BaseEstimator, TransformerMixin):
    """
    Process domain-specific features
    Example: Combine or modify features based on business logic
    """
    
    def __init__(self, variable_to_modify=None, variable_to_add=None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Add domain-specific transformations here
        # Example: Total income = ApplicantIncome + CoapplicantIncome
        if 'ApplicantIncome' in X.columns and 'CoapplicantIncome' in X.columns:
            X['TotalIncome'] = X['ApplicantIncome'] + X['CoapplicantIncome']
        
        return X


class MeanImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing numerical values with mean
    """
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables] if variables else []
        else:
            self.variables = variables
        self.imputer_dict_ = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                self.imputer_dict_[var] = X[var].mean()
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                X.fillna(self.imputer_dict_, inplace=True)

        return X


class ModeImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing categorical values with mode (most frequent)
    """
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables] if variables else []
        else:
            self.variables = variables
        self.imputer_dict_ = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                mode_value = X[var].mode()
                self.imputer_dict_[var] = mode_value[0] if len(mode_value) > 0 else None
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if var in X.columns and var in self.imputer_dict_:
                X.fillna(self.imputer_dict_, inplace=True)

        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from the dataframe
    """
    
    def __init__(self, variables_to_drop=None):
        if not isinstance(variables_to_drop, list):
            self.variables_to_drop = [variables_to_drop] if variables_to_drop else []
        else:
            self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Drop columns that exist
        cols_to_drop = [col for col in self.variables_to_drop if col in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        return X


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables to numerical
    """
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables] if variables else []
        else:
            self.variables = variables
        self.encoder_dict_ = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                unique_vals = X[var].dropna().unique()
                self.encoder_dict_[var] = {val: idx for idx, val in enumerate(unique_vals)}
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if var in X.columns and var in self.encoder_dict_:
                # Map known values, keep NaN for unknown
                X[var] = X[var].map(self.encoder_dict_[var])
        return X


class LogTransforms(BaseEstimator, TransformerMixin):
    """
    Apply log transformation to specified variables
    Useful for right-skewed distributions
    """
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables] if variables else []
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                # Add 1 to avoid log(0), handle negative values
                X[var] = np.log(X[var] + 1)
        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Group rare categories into a single 'Rare' category
    """
    
    def __init__(self, variables=None, tol=0.05):
        if not isinstance(variables, list):
            self.variables = [variables] if variables else []
        else:
            self.variables = variables
        self.tol = tol  # Threshold for rare labels
        self.encoder_dict_ = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                # Find frequent labels
                freq = X[var].value_counts() / len(X)
                frequent_labels = freq[freq >= self.tol].index.tolist()
                self.encoder_dict_[var] = frequent_labels
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if var in X.columns and var in self.encoder_dict_:
                # Replace rare labels with 'Rare'
                X[var] = X[var].apply(
                    lambda x: x if x in self.encoder_dict_[var] else 'Rare'
                )
        return X