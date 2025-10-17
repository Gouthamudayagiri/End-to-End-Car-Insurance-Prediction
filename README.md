# End-to-End-Car-Insurance-Prediction
# Insurance Charges Prediction

A machine learning project to predict insurance charges based on customer demographics and health information.

## Features

- **Data Ingestion**: From PostgreSQL database
- **Data Validation**: Schema validation and drift detection
- **Data Transformation**: Feature engineering and preprocessing
- **Model Training**: Automated model selection using neuro_mf
- **Model Evaluation**: Performance comparison and selection
- **Model Deployment**: FastAPI web application
- **Cloud Integration**: AWS S3 for model storage

## Project Structure



```
mlflow ui --backend-store-uri file:///./mlruns --port 5000
 http://localhost:5000 

 ````

 ```
     python test.py → Runs ML pipeline with MLflow tracking only

    dvc repro → Runs ML pipeline with DVC tracking + MLflow tracking
    dvc repro
    dvc repro && dvc push
````