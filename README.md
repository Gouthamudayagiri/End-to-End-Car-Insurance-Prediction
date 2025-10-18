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

    python test.py - ML pipeline with MLflow only

    dvc repro - Full pipeline with DVC + MLflow tracking

    python app.py - Start web application


    # View local experiments
mlflow ui --backend-store-uri file:///./mlruns --port 5000

# View S3 experiments (if configured)
mlflow ui --backend-store-uri s3://insurance-charges-model-2025/mlflow --port 5001

 mlflow ui --backend-store-uri file:///./mlruns

 mlflow ui --backend-store-uri file:///./mlruns --port 5000


 💡 View Results:
MLflow UI: mlflow ui --backend-store-uri file:///./mlruns --port 5000
DVC Status: python dvc_commands.py status
DVC Metrics: dvc metrics show
python dvc_commands.py push