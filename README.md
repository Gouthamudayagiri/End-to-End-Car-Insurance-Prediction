# 🚀 Insurance Charges Prediction - End-to-End MLOps System

A production-ready machine learning system for predicting insurance charges with full MLOps capabilities, CI/CD deployment, and cloud infrastructure.

## 📊 Project Overview

This project implements an end-to-end MLOps pipeline that:
- **Ingests data** from PostgreSQL database
- **Validates data quality** and detects drift
- **Engineers features** and transforms data
- **Trains multiple models** with automated selection
- **Tracks experiments** with MLflow
- **Versions data** with DVC
- **Deploys models** via FastAPI web interface
- **Stores artifacts** in AWS S3
- **Automates deployment** with CI/CD pipelines

## 🎯 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/End-to-End-Car-Insurance-Prediction.git
cd End-to-End-Car-Insurance-Prediction

# Create environment
conda create -n insurance python=3.8 -y
conda activate insurance

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
Create `.env` file:
```bash
# Database
POSTGRES_URL=postgresql://username:password@localhost:5432/car_insurance

# AWS
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1

# Application
MODEL_BUCKET_NAME=insurance-charges-model-2025
MLFLOW_TRACKING_URI=http://localhost:5000
```

## 🚀 Running the System

### Option 1: Complete Training Pipeline
```bash
python test.py
```

### Option 2: DVC Pipeline with Versioning
```bash
# Setup DVC (first time)
python dvc_commands.py setup

# Run pipeline
dvc repro
```

### Option 3: Start Web Application
```bash
python app.py
# Visit: http://localhost:8080
```

## 📊 Viewing Results

### MLflow Experiments
```bash
mlflow ui --backend-store-uri file:///./mlruns --port 5000
# Visit: http://localhost:5000
```

### DVC Metrics
```bash
dvc metrics show
python dvc_commands.py status
```

## 🏗️ Architecture

```
PostgreSQL → Data Ingestion → Validation → Transformation
                                      ↓
MLflow EC2 (Port 5000) ← Model Training → FastAPI EC2 (Port 8080) → Users
                                      ↓
DVC Versioning → S3 Storage → GitHub Actions CI/CD
```

## ☁️ AWS Production Deployment

### MLflow Server Setup (Separate EC2)

#### Step 1: Create MLflow EC2 Instance
- **AMI**: Ubuntu 22.04 LTS
- **Instance Type**: t2.micro
- **Security Group**: Open port 5000 (0.0.0.0/0)

#### Step 2: Install and Start MLflow
```bash
# Connect to MLflow EC2
ssh -i your-key.pem ubuntu@mlflow-ec2-ip

# Install dependencies
sudo apt update -y
sudo apt install python3-pip -y
pip3 install mlflow boto3 awscli

# Configure AWS
aws configure

# Start MLflow server
mlflow server \
    -h 0.0.0.0 \
    -p 5000 \
    --default-artifact-root s3://insurance-charges-model-2025 \
    --backend-store-uri sqlite:///mlflow.db \
    --serve-artifacts \
    --allowed-hosts "*"
```

**Access MLflow UI**: `http://your-mlflow-ec2-ip:5000`

### Application Deployment with CI/CD

#### Step 1: AWS IAM User Setup
1. **Login to AWS Console**
2. **Create IAM user for deployment** with specific access:
   - **EC2 access**: For virtual machine management
   - **ECR access**: Elastic Container Registry for Docker images

#### Step 2: Required IAM Policies
Attach these policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`
- `AmazonS3FullAccess`

#### Step 3: Create ECR Repository
```bash
aws ecr create-repository --repository-name insurance-predictor --region us-east-1
```
**Save ECR URI**: `123456789012.dkr.ecr.us-east-1.amazonaws.com/insurance-predictor`

#### Step 4: Create Application EC2 Instance
1. **Launch EC2 instance** (Ubuntu 22.04 LTS)
2. **Security Group**: Open ports 22 (SSH), 8080 (App)
3. **Instance Type**: t2.medium or larger for ML workloads

#### Step 5: Install Docker on Application EC2
```bash
# Connect to Application EC2
ssh -i your-key.pem ubuntu@app-ec2-ip

# Install Docker
sudo apt update -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

#### Step 6: Configure GitHub Self-Hosted Runner
1. Go to **GitHub Repository → Settings → Actions → Runners**
2. Click **New self-hosted runner**
3. Select **Linux** and follow the setup commands on your EC2 instance

#### Step 7: Setup GitHub Secrets
Add these secrets in your GitHub repository:

| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS Secret Key |
| `AWS_REGION` | `us-east-1` |
| `MLFLOW_TRACKING_URI` | `http://your-mlflow-ec2-ip:5000` |
| `MODEL_BUCKET_NAME` | `insurance-charges-model-2025` |
| `POSTGRES_URL` | Your PostgreSQL connection string |

### CI/CD Pipeline Workflow

Create `.github/workflows/cicd-pipeline.yml`:
```yaml
name: 🚀 Deploy Insurance ML System

on:
  push:
    branches: [ main ]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: insurance-predictor

jobs:
  build-and-deploy:
    runs-on: self-hosted
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
    
    - name: Build and push docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to EC2
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker stop insurance-app || true
        docker rm insurance-app || true
        docker pull $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker run -d \
          --name insurance-app \
          -p 8080:8080 \
          -e AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }} \
          -e AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }} \
          -e AWS_REGION=${{ env.AWS_REGION }} \
          -e MODEL_BUCKET_NAME=${{ secrets.MODEL_BUCKET_NAME }} \
          -e POSTGRES_URL=${{ secrets.POSTGRES_URL }} \
          -e MLFLOW_TRACKING_URI=${{ secrets.MLFLOW_TRACKING_URI }} \
          $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
```

## 📈 Model Performance

- **Best Model**: XGBoost (Automatically selected via GridSearch)
- **Accuracy**: 88.4% R² Score
- **Features**: 12 engineered features
- **Training Data**: 1,070 records
- **Testing Data**: 268 records

## 🛠️ Technology Stack

- **ML Frameworks**: Scikit-learn, XGBoost, Pandas, NumPy
- **Experiment Tracking**: MLflow (Separate EC2)
- **Data Versioning**: DVC
- **Cloud Infrastructure**: AWS EC2, ECR, S3
- **Database**: PostgreSQL
- **Web Framework**: FastAPI
- **CI/CD**: GitHub Actions with Self-Hosted Runner
- **Containerization**: Docker

## 🚀 Production URLs

After deployment:
- **Application**: `http://your-app-ec2-ip:8080`
- **MLflow UI**: `http://your-mlflow-ec2-ip:5000`
- **Health Check**: `http://your-app-ec2-ip:8080/health`

## 📝 Key Features

- **Automated Model Selection**: GridSearchCV for best model
- **Dual EC2 Setup**: Separate instances for MLflow and Application
- **Health Monitoring**: Built-in health checks and MLflow connection testing
- **S3 Integration**: All artifacts stored in AWS S3
- **CI/CD Automation**: Zero-downtime deployments with GitHub Actions

Push to main branch triggers automatic deployment to production! 🚀