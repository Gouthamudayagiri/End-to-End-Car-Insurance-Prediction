# setup.py - FIX PACKAGING
from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="insurance_charges",
    version="1.0.0",
    package_dir={"": "src"},  # CRITICAL: Tell setuptools where packages are
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'insurance-train=insurance_charges.pipeline.training_pipeline:TrainPipeline.run_pipeline',
        ],
    },
)