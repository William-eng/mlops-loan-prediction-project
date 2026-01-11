#!/bin/sh
set -e

MODEL_PATH=/app/prediction_model/trained_models/model.pkl

# Check if model exists (bundled in image or from previous download)
if [ -f "$MODEL_PATH" ]; then
  echo "✓ Model found at $MODEL_PATH"
  python -c "from pathlib import Path; p = Path('$MODEL_PATH'); print(f'  Model size: {p.stat().st_size / 1024:.2f} KB')"
  
# If not found, try downloading from Azure (if connection string provided)
elif [ -n "$AZURE_STORAGE_CONNECTION_STRING" ]; then
  echo "==> Model not found locally, downloading from Azure Blob"
  
  python << 'EOF'
import os
from azure.storage.blob import BlobClient
from pathlib import Path

try:
    # Ensure directory exists
    model_dir = Path("/app/prediction_model/trained_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download blob
    print("Connecting to Azure Blob Storage...")
    blob = BlobClient.from_connection_string(
        os.environ['AZURE_STORAGE_CONNECTION_STRING'], 
        container_name='loanprediction', 
        blob_name='model.pkl'
    )
    
    print("Downloading model...")
    model_path = model_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        blob_data = blob.download_blob()
        f.write(blob_data.readall())
    
    print(f"✓ Model downloaded successfully to {model_path}")
    print(f"  Model size: {model_path.stat().st_size / 1024:.2f} KB")
    
except Exception as e:
    print(f"✗ Error downloading model: {e}")
    print("  Please check:")
    print("  - AZURE_STORAGE_CONNECTION_STRING is set correctly")
    print("  - Container 'loanprediction' exists")
    print("  - Blob 'model.pkl' exists in the container")
    exit(1)
EOF

else
  echo "✗ Error: Model not found and no Azure connection string provided"
  echo "  Please either:"
  echo "  - Build the image with the model included (COPY models/loan_model.pkl)"
  echo "  - Provide AZURE_STORAGE_CONNECTION_STRING environment variable"
  exit 1
fi

echo ""
echo "==> Starting FastAPI application"
exec python main.py