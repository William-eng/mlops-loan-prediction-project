"""
Diagnose config issue
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("DIAGNOSING CONFIG ISSUE")
print("=" * 70)

# Import config
from prediction_model.config import config

# Check what config actually is
print(f"\n1️⃣ Config module location:")
print(f"   {config.__file__}")

print(f"\n2️⃣ DATAPATH attribute:")
print(f"   Value: {config.DATAPATH}")
print(f"   Type: {type(config.DATAPATH)}")

print(f"\n3️⃣ Check if it's a string:")
if isinstance(config.DATAPATH, str):
    print(f"   ❌ DATAPATH is a STRING (should be Path)")
    print(f"   Converting to Path...")
    from pathlib import Path as P
    datapath = P(config.DATAPATH)
    print(f"   Converted: {datapath}")
    print(f"   Type: {type(datapath)}")
else:
    print(f"   ✓ DATAPATH is already a Path object")

print("\n" + "=" * 70)

# List all config files
print("\nSearching for config files:")
import os
for root, dirs, files in os.walk('prediction_model'):
    for file in files:
        if 'config' in file and file.endswith('.py'):
            filepath = os.path.join(root, file)
            print(f"   Found: {filepath}")