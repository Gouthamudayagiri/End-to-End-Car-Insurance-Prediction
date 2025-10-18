# app.py - UPDATED FOR PRODUCTION DEPLOYMENT
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional

import os
import sys

# Add current directory to path for proper imports - FIXED PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import MLflow and set tracking URI early
import mlflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"🎯 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

try:
    from src.insurance_charges.constants import APP_HOST, APP_PORT
    from src.insurance_charges.pipeline.prediction_pipeline import InsuranceData, InsuranceClassifier
    from src.insurance_charges.pipeline.training_pipeline import TrainPipeline
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Provide fallbacks for critical constants
    APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
    APP_PORT = int(os.getenv('APP_PORT', '8080'))

app = FastAPI(
    title="Insurance Charges Prediction API",
    description="ML API for predicting insurance charges based on customer demographics",
    version="1.0.0"
)

# Mount static files with error handling
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory='templates')
    print("✅ Static files mounted successfully")
except Exception as e:
    print(f"⚠️ Static files setup warning: {e}")
    # Create fallback templates object
    templates = Jinja2Templates(directory='.')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InsuranceForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.age: Optional[str] = None
        self.sex: Optional[str] = None
        self.bmi: Optional[str] = None
        self.children: Optional[str] = None
        self.smoker: Optional[str] = None
        self.region: Optional[str] = None

    async def get_insurance_data(self):
        form = await self.request.form()
        self.age = form.get("age")
        self.sex = form.get("sex")
        self.bmi = form.get("bmi")
        self.children = form.get("children")
        self.smoker = form.get("smoker")
        self.region = form.get("region")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "insurance.html", 
        {"request": request, "context": "Enter your details to predict insurance charges"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return {"status": "healthy", "service": "insurance-predictor"}

@app.get("/train")
async def trainRouteClient():
    try:
        print("🚀 Starting training pipeline...")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        print(f"❌ Training error: {e}")
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = InsuranceForm(request)
        await form.get_insurance_data()
        
        # Validate required fields
        if not all([form.age, form.sex, form.bmi, form.children, form.smoker, form.region]):
            return templates.TemplateResponse(
                "insurance.html",
                {"request": request, "context": "Please fill all fields!"},
            )
        
        # Validate data types
        try:
            insurance_data = InsuranceData(
                age=int(form.age),
                sex=form.sex,
                bmi=float(form.bmi),
                children=int(form.children),
                smoker=form.smoker,
                region=form.region
            )
        except ValueError as e:
            return templates.TemplateResponse(
                "insurance.html",
                {"request": request, "context": f"Invalid input data: {str(e)}"},
            )
        
        insurance_df = insurance_data.get_insurance_input_data_frame()
        model_predictor = InsuranceClassifier()
        predicted_charges = model_predictor.predict(dataframe=insurance_df)[0]
        
        # Format the prediction
        formatted_charges = f"${float(predicted_charges):,.2f}"
        
        print(f"✅ Prediction successful: {formatted_charges}")
        
        return templates.TemplateResponse(
            "insurance.html",
            {"request": request, "context": f"Predicted Insurance Charges: {formatted_charges}"},
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return templates.TemplateResponse(
            "insurance.html",
            {"request": request, "context": f"Error: {str(e)}"},
        )

@app.get("/env")
async def show_environment():
    """Debug endpoint to show environment variables (remove in production)"""
    env_vars = {
        "MLFLOW_TRACKING_URI": os.getenv('MLFLOW_TRACKING_URI'),
        "MODEL_BUCKET_NAME": os.getenv('MODEL_BUCKET_NAME'),
        "AWS_REGION": os.getenv('AWS_DEFAULT_REGION'),
        "APP_HOST": APP_HOST,
        "APP_PORT": APP_PORT
    }
    return env_vars

if __name__ == "__main__":
    print(f"🚀 Starting Insurance Prediction API on {APP_HOST}:{APP_PORT}")
    print(f"📊 MLflow Tracking: {MLFLOW_TRACKING_URI}")
    app_run(app, host=APP_HOST, port=APP_PORT)