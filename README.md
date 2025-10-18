
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
MLflow Tracking ← Model Training → Evaluation → FastAPI → AWS Deployment
                                      ↓
DVC Versioning → S3 Storage → Model Registry → CI/CD Pipeline
```

## ☁️ AWS CI/CD Deployment

### Step 1: AWS IAM User Setup
1. **Login to AWS Console**
2. **Create IAM user for deployment** with specific access:
   - **EC2 access**: For virtual machine management
   - **ECR access**: Elastic Container Registry for Docker images

### Step 2: Required IAM Policies
Attach the following policies to your IAM user:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "ecr:*",
        "s3:*",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

**Specific Policies Needed:**
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`
- `AmazonS3FullAccess`

### Step 3: Create ECR Repository
```bash
# Create ECR repo to store Docker image
aws ecr create-repository --repository-name insurance-predictor --region us-east-1

# Save the URI: 123456789012.dkr.ecr.us-east-1.amazonaws.com/insurance-predictor
```

### Step 4: Create EC2 Instance
1. **Launch EC2 instance** (Ubuntu 20.04 LTS)
2. **Security Group**: Open ports 22 (SSH), 80 (HTTP), 8080 (App)
3. **Instance Type**: t2.medium or larger for ML workloads

### Step 5: Install Docker on EC2
```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@ec2-ip-address

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu
newgrp docker

# Verify installation
docker --version
```

### Step 6: Configure GitHub Self-Hosted Runner
1. Go to **GitHub Repository → Settings → Actions → Runners**
2. Click **New self-hosted runner**
3. Select **Linux** and follow the setup commands on your EC2 instance

### Step 7: Setup GitHub Secrets
Add these secrets in your GitHub repository (**Settings → Secrets → Actions**):

| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS Secret Key |
| `AWS_REGION` | `us-east-1` |
| `AWS_ECR_LOGIN_URI` | `123456789012.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | `insurance-predictor` |
| `MODEL_BUCKET_NAME` | `insurance-charges-model-2025` |

## 🔧 CI/CD Pipeline Workflow

The GitHub Actions workflow automatically:
1. **Builds Docker image** of the source code
2. **Pushes Docker image** to ECR
3. **Deploys to EC2** instance
4. **Pulls image** from ECR in EC2
5. **Launches container** with the application

### GitHub Actions Workflow File
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to AWS EC2

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: self-hosted
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ secrets.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build, tag, and push image to Amazon ECR
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY_NAME }}
        IMAGE_TAG: latest
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
    
    - name: Deploy to EC2
      run: |
        docker pull ${{ steps.login-ecr.outputs.registry }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
        docker stop insurance-app || true
        docker rm insurance-app || true
        docker run -d -p 8080:8080 --name insurance-app \
          -e AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }} \
          -e AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }} \
          -e MODEL_BUCKET_NAME=${{ secrets.MODEL_BUCKET_NAME }} \
          ${{ steps.login-ecr.outputs.registry }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "app.py"]
```

### Build and Run Locally
```bash
# Build image
docker build -t insurance-predictor .

# Run container
docker run -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  insurance-predictor
```

## 📈 Model Performance

- **Best Model**: XGBoost
- **Accuracy**: 88.4% R² Score
- **Features**: 12 engineered features
- **Training Data**: 1,070 records
- **Testing Data**: 268 records
- **Cloud Storage**: AWS S3 with 39 MLflow artifacts

## 🔧 Advanced Usage

### Run Specific Pipeline Stages
```bash
python dvc_pipeline.py --stage data_ingestion
python dvc_pipeline.py --stage model_training
python dvc_pipeline.py --stage full
```

### Manual S3 Operations
```bash
python dvc_commands.py push    # Push to S3
python dvc_commands.py pull    # Pull from S3
python dvc_commands.py status  # Check status
```

### Verify MLflow S3 Integration
```bash
python verify_mlflow_s3_fixed.py
python test_mlflow_s3_live.py
```

## 🛠️ Technology Stack

- **ML Frameworks**: Scikit-learn, XGBoost, Pandas, NumPy
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Cloud Infrastructure**: AWS EC2, ECR, S3
- **Database**: PostgreSQL
- **Web Framework**: FastAPI
- **Validation**: Evidently AI
- **CI/CD**: GitHub Actions
- **Containerization**: Docker

## 📁 Project Structure
```
End-to-End-Car-Insurance-Prediction/
├── src/insurance_charges/     # Source code
├── config/                    # Configuration files
├── artifacts/                 # Pipeline artifacts
├── mlruns/                   # MLflow experiments
├── data/                     # Processed data (DVC tracked)
├── models/                   # Trained models
├── reports/                  # Analysis reports
├── .github/workflows/        # CI/CD pipelines
├── Dockerfile               # Container configuration
└── docker-compose.yml       # Multi-container setup
```

## 🚀 Production Deployment

### Manual Deployment
```bash
# Build and push to ECR
docker build -t insurance-predictor .
docker tag insurance-predictor:latest your-ecr-url/insurance-predictor:latest
docker push your-ecr-url/insurance-predictor:latest

# Deploy to EC2
ssh ubuntu@your-ec2-ip "docker pull your-ecr-url/insurance-predictor:latest && docker run -d -p 8080:8080 your-ecr-url/insurance-predictor:latest"
```

### Automated Deployment
Push to main branch triggers automatic deployment via GitHub Actions.

#### mlflow depoyment 
```
MLflow on AWS
MLflow on AWS Setup:

    Login to AWS console.
    Create IAM user with AdministratorAccess
    Export the credentials in your AWS CLI by running "aws configure"
    Create a s3 bucket
    Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
```
````
sudo apt update

sudo apt install python3-pip

sudo pip3 install pipenv

sudo pip3 install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell


## Then set aws credentials
aws configure


#Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://insurance-charges-model-2025

#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-54-209-33-81.compute-1.amazonaws.com:5000/





mlflow server \
    -h 0.0.0.0 \
    -p 5000 \
    --default-artifact-root s3://insurance-charges-model-2025 \
    --backend-store-uri sqlite:///mlflow.db \
    --serve-artifacts \
    --allowed-hosts "*"
```

```
AWS-CICD-Deployment-with-Github-Actions
1. Login to AWS console.
2. Create IAM user for deployment

#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess

3. Create ECR repo to store/save docker image

- Save the URI: 136566696263.dkr.ecr.us-east-1.amazonaws.com/mlproject

4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

6. Configure EC2 as self-hosted runner:

setting>actions>runner>new self hosted runner> choose os> then run command one by one

7. Setup github secrets:

    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION
    ECR_REPO
```
