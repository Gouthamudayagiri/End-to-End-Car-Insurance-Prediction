# app.py - CAR INSURANCE VERSION (like your visa example)
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional

import os
import sys

# Add current directory to path for proper imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.insurance_charges.constants import APP_HOST, APP_PORT
from src.insurance_charges.pipeline.prediction_pipeline import InsuranceData, InsuranceClassifier
from src.insurance_charges.pipeline.training_pipeline import TrainPipeline

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

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")

    except Exception as e:
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
        predicted_charges = model_predictor.predict(dataframe=insurance_df)[0]
        
        # Format the prediction
        formatted_charges = f"${float(predicted_charges):,.2f}"
        
        return templates.TemplateResponse(
            "insurance.html",
            {"request": request, "context": f"Predicted Insurance Charges: {formatted_charges}"},
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "insurance.html",
            {"request": request, "context": f"Error: {str(e)}"},
        )

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)