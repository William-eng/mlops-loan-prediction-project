# """
# FastAPI application for serving predictions
# """

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# import uvicorn
# import pandas as pd
# import io
# from prediction_model.predict import generate_predictions, generate_predictions_batch
# from prediction_model.config import config
# import mlflow

# # Set MLflow URI
# mlflow.set_tracking_uri(config.TRACKING_URI)

# # Create FastAPI app
# app = FastAPI(
#     title="Loan Prediction API",
#     description="ML-powered loan approval prediction service",
#     version="1.0.0"
# )


# # Define request body structure
# class LoanApplication(BaseModel):
#     """
#     What data we expect from the user
#     """
#     Gender: str
#     Married: str
#     Dependents: str
#     Education: str
#     Self_Employed: str
#     ApplicantIncome: float
#     CoapplicantIncome: float
#     LoanAmount: float
#     Loan_Amount_Term: float
#     Credit_History: float
#     Property_Area: str
    
#     class Config:
#         # Example for documentation
#         json_schema_extra = {
#             "example": {
#                 "Gender": "Male",
#                 "Married": "Yes",
#                 "Dependents": "0",
#                 "Education": "Graduate",
#                 "Self_Employed": "No",
#                 "ApplicantIncome": 5000,
#                 "CoapplicantIncome": 2000,
#                 "LoanAmount": 150,
#                 "Loan_Amount_Term": 360,
#                 "Credit_History": 1.0,
#                 "Property_Area": "Urban"
#             }
#         }


# @app.get("/")
# def home():
#     """
#     Welcome endpoint - just to check if API is running
#     """
#     return {
#         "message": "🏦 Loan Prediction API",
#         "status": "Running",
#         "endpoints": {
#             "/docs": "API documentation",
#             "/predict": "Single prediction",
#             "/batch_predict": "Batch predictions (CSV upload)"
#         }
#     }


# @app.get("/health")
# def health_check():
#     """
#     Health check endpoint - important for Kubernetes
#     """
#     return {"status": "healthy"}


# @app.post("/predict")
# def predict(loan_app: LoanApplication):
#     """
#     Single prediction endpoint
    
#     Send one loan application, get approval decision
#     """
#     try:
#         # Convert Pydantic model to dictionary
#         data = loan_app.model_dump()
        
#         # Make prediction
#         prediction = generate_predictions([data])["prediction"][0]
        
#         # Convert to readable format
#         status = "Approved ✓" if prediction == "Y" else "Rejected ✗"
        
#         return {
#             "status": status,
#             "prediction": prediction,
#             "applicant_income": loan_app.ApplicantIncome,
#             "loan_amount": loan_app.LoanAmount
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# @app.post("/batch_predict")
# async def batch_predict(file: UploadFile = File(...)):
#     """
#     Batch prediction endpoint
    
#     Upload a CSV file with multiple loan applications
#     Returns CSV with predictions added
#     """
#     try:
#         # Read uploaded CSV
#         content = await file.read()
#         df = pd.read_csv(io.BytesIO(content), index_col=False)
        
#         print(f"📄 Received CSV with {len(df)} rows")
        
#         # Validate columns
#         required_columns = config.FEATURES
#         missing_columns = [col for col in required_columns if col not in df.columns]
        
#         if missing_columns:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Missing columns: {missing_columns}"
#             )
        
#         # Make predictions
#         predictions = generate_predictions_batch(df[required_columns])["prediction"]
        
#         # Add predictions to dataframe
#         df['Prediction'] = predictions
#         df['Status'] = df['Prediction'].map({'Y': 'Approved', 'N': 'Rejected'})
        
#         # Convert back to CSV
#         result = df.to_csv(index=False)
        
#         # Return as downloadable file
#         return StreamingResponse(
#             io.BytesIO(result.encode('utf-8')),
#             media_type="text/csv",
#             headers={"Content-Disposition": "attachment; filename=predictions.csv"}
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# if __name__ == "__main__":
#     # Run with: python main.py
#     print("="*70)
#     print("🚀 Starting Loan Prediction API")
#     print("="*70)
#     print("📍 URL: http://localhost:8005")
#     print("📚 Docs: http://localhost:8005/docs")
#     print("="*70)
    
#     uvicorn.run(app, host="0.0.0.0", port=8005)

"""
FastAPI application for serving loan predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import io
import os
from pathlib import Path
from prediction_model.predict import generate_predictions, generate_predictions_batch
from prediction_model.config import config
from azure.storage.blob import BlobClient

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

app = FastAPI(...)

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Custom metrics
prediction_counter = Counter(
    'loan_predictions_total', 
    'Total loan predictions',
    ['prediction_result']
)

prediction_duration = Histogram(
    'loan_prediction_duration_seconds',
    'Prediction duration',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

model_loaded = Gauge('model_loaded', 'Model load status')


# Paths
MODEL_DIR = Path(__file__).parent / "prediction_model" / "trained_models"
MODEL_FILE = MODEL_DIR / "model.pkl"

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Download model from Azure Blob if not present
AZURE_CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
if not MODEL_FILE.exists():
    if not AZURE_CONN:
        raise RuntimeError(
            "Azure connection string not provided. Set AZURE_STORAGE_CONNECTION_STRING env variable."
        )
    print("==> Downloading model from Azure Blob storage...")
    blob = BlobClient.from_connection_string(
        conn_str=AZURE_CONN,
        container_name="loanprediction",
        blob_name="model.pkl"
    )
    with open(MODEL_FILE, "wb") as f:
        f.write(blob.download_blob().readall())
    print("✅ Model downloaded successfully!")

# Create FastAPI app
app = FastAPI(
    title="Loan Prediction API",
    description="ML-powered loan approval prediction service",
    version="1.0.0"
)

# ---------------------------
# Request schema
# ---------------------------
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 2000,
                "LoanAmount": 150,
                "Loan_Amount_Term": 360,
                "Credit_History": 1.0,
                "Property_Area": "Urban"
            }
        }

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def home():
    return {
        "message": "🏦 Loan Prediction API",
        "status": "Running",
        "endpoints": {
            "/docs": "API documentation",
            "/predict": "Single prediction",
            "/batch_predict": "Batch predictions (CSV upload)"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(loan_app: LoanApplication):
    try:
        data = loan_app.model_dump()
        prediction = generate_predictions([data])["prediction"][0]
        status = "Approved ✓" if prediction == "Y" else "Rejected ✗"
        return {
            "status": status,
            "prediction": prediction,
            "applicant_income": loan_app.ApplicantIncome,
            "loan_amount": loan_app.LoanAmount
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), index_col=False)
        required_columns = config.FEATURES
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_columns}"
            )
        predictions = generate_predictions_batch(df[required_columns])["prediction"]
        df["Prediction"] = predictions
        df["Status"] = df["Prediction"].map({"Y": "Approved", "N": "Rejected"})
        result = df.to_csv(index=False)
        return StreamingResponse(
            io.BytesIO(result.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("="*70)
    print("🚀 Starting Loan Prediction API")
    print("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8005)
