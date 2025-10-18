# setup.py
from setuptools import setup, find_packages

def read_requirements():
    """Read requirements from requirements.txt"""
    requirements = []
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith(('#', '-e'))]
    except FileNotFoundError:
        print("⚠️ requirements.txt not found, using default requirements")
        requirements = [
            "scikit-learn>=1.2.0",
            "pandas>=1.5.0", 
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "xgboost>=1.6.0",
            "sqlalchemy>=1.4.0",
            "psycopg2-binary>=2.9.0",
            "boto3>=1.26.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.21.0",
            "python-multipart>=0.0.5",
            "jinja2>=3.0.0",
            "evidently>=0.2.8",
            "python-dotenv>=0.19.0",
            "pyyaml>=6.0",
            "dill>=0.3.0",
        ]
    return requirements

setup(
    name="insurance-charges",
    version="1.0.0",
    author="Goutham",
    author_email="your-email@example.com",
    description="ML Pipeline for Insurance Charges Prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    python_requires=">=3.8",
    keywords="machine-learning mlops insurance prediction",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)