"""
Test imports step by step
"""
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("STEP-BY-STEP IMPORT TEST")
print("=" * 70)

# Step 1
print("\n1️⃣ Import config...")
try:
    from prediction_model.config import config
    print("   ✓ Config imported")
    print(f"   - DATAPATH: {config.DATAPATH}")
    print(f"   - TRAIN_FILE: {config.TRAIN_FILE}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Step 2
print("\n2️⃣ Import data_handling...")
try:
    from prediction_model.processing.data_handling import load_dataset
    print("   ✓ data_handling imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3
print("\n3️⃣ Import preprocessing classes...")
try:
    from prediction_model.processing.preprocessing import (
        DomainProcessing,
        MeanImputer,
        ModeImputer,
        DropColumns,
        CustomLabelEncoder,
        LogTransforms
    )
    print("   ✓ All preprocessing classes imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4
print("\n4️⃣ Check data file exists...")
data_file = config.DATAPATH / config.TRAIN_FILE
if data_file.exists():
    print(f"   ✓ Data file found: {data_file}")
else:
    print(f"   ✗ Data file NOT found: {data_file}")
    print(f"   Please ensure train.csv is in the data/ folder")

print("\n" + "=" * 70)
print("✅ ALL IMPORTS SUCCESSFUL!")
print("=" * 70)
print("\nYou can now run: python run_training.py")