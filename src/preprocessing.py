"""
Custom preprocessing transformers
These are like LEGO blocks we can stack together
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MeanImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing numbers with the average
    
    Example: If you have [1, 2, ?, 4], average is (1+2+4)/3 = 2.33
    Result: [1, 2, 2.33, 4]
    """
    def __init__(self, variables=None):
        self.variables = variables  # Which columns to fill
    
    def fit(self, X, y=None):
        # Learn the averages from training data
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self
    
    def transform(self, X):
        # Apply the learned averages
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mean_dict[col], inplace=True)
        return X


class ModeImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing categories with the most common value
    
    Example: If you have [Male, Female, ?, Male, Male]
    Most common is Male
    Result: [Male, Female, Male, Male, Male]
    """
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        # Learn the most common values
        self.mode_dict = {}
        for col in self.variables:
            self.mode_dict[col] = X[col].mode()[0]
        return self
    
    def transform(self, X):
        # Apply the most common values
        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.mode_dict[col], inplace=True)
        return X


class DomainProcessing(BaseEstimator, TransformerMixin):
    """
    Combines Applicant + Coapplicant income into total household income
    
    This makes sense because banks care about TOTAL household income
    """
    def __init__(self, variable_to_modify=None, variable_to_add=None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variable_to_modify:
            # Add coapplicant income to applicant income
            X[feature] = X[feature] + X[self.variable_to_add]
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Removes columns we don't need anymore
    """
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.variables_to_drop)
        return X


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Converts categories to numbers
    
    Example: Gender [Male, Female, Male] → [0, 1, 0]
    The order is based on frequency (most common = highest number)
    """
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.label_dict = {}
        for var in self.variables:
            # Sort by frequency
            t = X[var].value_counts().sort_values(ascending=True).index
            # Create mapping
            self.label_dict[var] = {k: i for i, k in enumerate(t, 0)}
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X


class LogTransforms(BaseEstimator, TransformerMixin):
    """
    Applies log transformation to reduce skewness
    
    Why? Income and loan amounts are often skewed (few very high values)
    Log transformation makes the distribution more normal
    
    Example: [10, 100, 1000] → [2.3, 4.6, 6.9] (more spread out evenly)
    """
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X