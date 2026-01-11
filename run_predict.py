"""
Run predictions using the trained model
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("=" * 70)
    print("STARTING PREDICTION")
    print("=" * 70)
    
    try:
        from prediction_model import predict
        print("\n✅ Prediction completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()