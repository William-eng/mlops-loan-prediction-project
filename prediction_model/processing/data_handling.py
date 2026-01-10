"""
Helper functions for loading data
"""

import os
import pandas as pd
from prediction_model.config import config

def load_dataset(file_name):
    """
    Load dataset from the datasets folder
    
    Args:
        file_name: Name of CSV file (e.g., 'train.csv')
    
    Returns:
        pandas DataFrame
    """
    filepath = os.path.join(config.DATAPATH, file_name)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    _data = pd.read_csv(filepath)
    print(f"✓ Loaded {len(_data)} rows from {file_name}")
    
    return _data