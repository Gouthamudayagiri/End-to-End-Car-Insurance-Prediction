from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run

from typing import Optional
import os

import sys


# Add src to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from insurance_charges.constants import APP_HOST, APP_PORT
from insurance_charges.pipeline.prediction_pipeline import InsuranceData, InsuranceClassifier
from insurance_charges.pipeline.training_pipeline import TrainPipeline
from insurance_charges.utils.main_utils import validate_environment_variables

app = FastAPI(
    title="Insurance Charges Prediction API",
    description="ML API for predicting insurance charges based on customer demographics",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Validate environment variables on startup"""
    try:
        validate_environment_variables()
        logging.info("Application started successfully")
    except Exception as e:
        logging.error(f"Startup validation failed: {e}")
        # Don't raise here to allow the app to start for debugging

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model can be loaded
        predictor = InsuranceClassifier()
        
        return {
            "status": "healthy",
            "message": "Insurance Charges Prediction API is running",
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        "insurance.html", 
        {"request": request, "context": "Rendering"}
    )

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return JSONResponse(
            content={"status": "success", "message": "Training completed successfully"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Training failed: {str(e)}"}
        )

@app.post("/predict")
async def predict_insurance_charges(request: Request):
    try:
        form = DataForm(request)
        await form.get_insurance_data()
        
        # Validate input data
        if not all([form.age, form.sex, form.bmi, form.children, form.smoker, form.region]):
            raise HTTPException(status_code=400, detail="All fields are required")
        
        insurance_data = InsuranceData(
            age=int(form.age),
            sex=form.sex,
            bmi=float(form.bmi),
            children=int(form.children),
            smoker=form.smoker,
            region=form.region
        )
        
        insurance_df = insurance_data.get_insurance_input_data_frame()
        model_predictor = InsuranceClassifier()
        predicted_charges = model_predictor.predict(dataframe=insurance_df)
        
        # Format the prediction result
        formatted_charges = f"${predicted_charges:,.2f}" if predicted_charges > 0 else "$0.00"
        
        return JSONResponse(
            content={
                "status": "success",
                "prediction": float(predicted_charges),
                "formatted_prediction": formatted_charges,
                "input_data": insurance_data.get_insurance_data_as_dict()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/")
async def predictRouteClient(request: Request):
    """Legacy endpoint for form submissions"""
    try:
        form = DataForm(request)
        await form.get_insurance_data()
        
        insurance_data = InsuranceData(
            age=int(form.age),
            sex=form.sex,
            bmi=float(form.bmi),
            children=int(form.children),
            smoker=form.smoker,
            region=form.region
        )
        
        insurance_df = insurance_data.get_insurance_input_data_frame()
        model_predictor = InsuranceClassifier()
        predicted_charges = model_predictor.predict(dataframe=insurance_df)
        
        formatted_charges = f"${predicted_charges:,.2f}" if predicted_charges > 0 else "$0.00"
        
        return templates.TemplateResponse(
            "insurance.html",
            {"request": request, "context": f"Predicted Insurance Charges: {formatted_charges}"},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)