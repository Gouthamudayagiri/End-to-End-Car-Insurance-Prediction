import os
from pathlib import Path

project_name = "src.insurance_charges"

list_of_files = [
    # Source code in src directory
    f"src/{project_name}/__init__.py",
    
    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",  
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",
    f"src/{project_name}/components/data_analysis.py",
    f"src/{project_name}/components/data_quality.py",
    
    # Configuration
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/postgres_db_connection.py",
    f"src/{project_name}/configuration/aws_connection.py",
    
    # Constants
    f"src/{project_name}/constants/__init__.py",
    
    # Entity
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",
    f"src/{project_name}/entity/estimator.py",
    f"src/{project_name}/entity/s3_estimator.py",
    
    # Exception
    f"src/{project_name}/exception/__init__.py",
    
    # Logger
    f"src/{project_name}/logger/__init__.py",
    
    # Pipeline
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    
    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    f"src/{project_name}/utils/retry_utils.py",
    
    # Data Access
    f"src/{project_name}/data_access/__init__.py",
    f"src/{project_name}/data_access/insurance_data.py",
    
    # Cloud Storage
    f"src/{project_name}/cloud_storage/__init__.py",
    f"src/{project_name}/cloud_storage/aws_storage.py",
    
    # Frontend
    "static/css/style.css",
    "templates/insurance.html",
    
    # Configuration files
    "config/model.yaml",
    "config/schema.yaml",
    "config/hyperparameters.yaml",
    
    # Application files
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
    "Makefile",
    
    # Documentation
    "README.md",
    "DATABASE_SETUP.md",
    
    # Environment & Setup
    ".env.example",
    "setup.py",
    # "pyproject.toml",
    
    # Testing
    "tests/__init__.py",
    "tests/test_data_validation.py",
    "tests/test_model_training.py",
    
    # Notebooks
    "notebooks/exploratory_analysis.ipynb",
    
    # CI/CD
    ".github/workflows/ci-cd.yml",
    
    # Git
    ".gitignore"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        print(f"Created: {filepath}")
    else:
        print(f"File already present at: {filepath}")

print("\n✅ Project structure created successfully!")
