# """
# Run predictions using the trained model
# """
# import sys
# from pathlib import Path

# # Add project root to path
# sys.path.insert(0, str(Path(__file__).parent))

# if __name__ == "__main__":
#     print("=" * 70)
#     print("STARTING PREDICTION")
#     print("=" * 70)
    
#     try:
#         from prediction_model import predict
#         print("\n✅ Prediction completed successfully!")
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()

"""
Run predictions using the trained model (inference-only)
"""

import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Paths
MODEL_DIR = Path(__file__).parent / "prediction_model" / "trained_models"
MODEL_FILE = MODEL_DIR / "model.pkl"

# Ensure model exists
AZURE_CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
if not MODEL_FILE.exists():
    if not AZURE_CONN:
        raise RuntimeError(
            "Azure connection string not provided. Set AZURE_STORAGE_CONNECTION_STRING env variable."
        )
    print("==> Downloading model from Azure Blob storage...")
    from azure.storage.blob import BlobClient
    blob = BlobClient.from_connection_string(
        conn_str=AZURE_CONN,
        container_name="loanprediction",
        blob_name="model.pkl"
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        f.write(blob.download_blob().readall())
    print("✅ Model downloaded successfully!")

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Running Prediction Script")
    print("=" * 70)

    try:
        from prediction_model import predict
        print("✅ Prediction module loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading prediction module: {e}")
        import traceback
        traceback.print_exc()
