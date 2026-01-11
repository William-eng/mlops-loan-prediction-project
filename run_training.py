"""
Run the training pipeline
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import
if __name__ == "__main__":
    from prediction_model import training_pipeline