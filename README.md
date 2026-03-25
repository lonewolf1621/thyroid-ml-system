# Thyroid ML System

This project is an end-to-end machine learning system for predicting thyroid disease using both structured clinical data and medical images. The goal was to move beyond a simple model and build something closer to a real-world ML application, including training pipelines, APIs, monitoring, and deployment.

---

## Overview

The system supports two types of inputs:

* Tabular clinical data (such as lab values and patient information)
* Medical images for classification (benign vs malignant)

For tabular data, multiple machine learning models were trained and compared. For images, a deep learning model based on transfer learning was used.

The project also includes model tracking, versioning, logging, and a deployable API.

---

## Project Structure

Thyroid-ML-System/

* api/ → FastAPI application for serving predictions
* src/ → Training scripts for tabular and image models
* monitoring/ → Drift detection logic
* models/ → Saved tabular models (versioned)
* image_models/ → Saved deep learning model
* logs/ → Prediction logs
* .github/workflows/ → CI/CD pipeline
* Dockerfile → Container setup
* requirements.txt → Dependencies

---

## What I Built

* A preprocessing pipeline using scikit-learn (handling missing values, scaling, encoding)
* Multiple models for comparison (RandomForest, LogisticRegression, XGBoost)
* A CNN-based image classifier using EfficientNet with transfer learning
* A FastAPI service with endpoints for both tabular and image predictions
* Model versioning to manage multiple trained models
* Experiment tracking using MLflow
* Prediction logging for monitoring
* Basic drift detection using Evidently
* Docker setup for deployment
* CI/CD pipeline using GitHub Actions

---

## How to Run the Project

Clone the repository:

git clone https://github.com/lonewolf1621/thyroid-ml-system.git
cd thyroid-ml-system

Create a virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the API:

python -m uvicorn api.main:app --reload

Then open:

http://127.0.0.1:8000/docs

---

## API Endpoints

POST /predict
Used for predictions on tabular data.

POST /predict-image
Used for image classification.

---

## Notes

The dataset and image files are not included in this repository.
To run training locally, place your data in the `data/` folder and images in the `images/` folder.

---

## Key Takeaways

This project helped me understand how to:

* Structure a machine learning project beyond notebooks
* Work with both tabular and image data in a single system
* Build and serve models using FastAPI
* Track experiments and manage model versions
* Handle basic monitoring and drift detection
* Package everything using Docker and automate workflows with CI/CD

---

## Future Improvements

* Automate model retraining when drift is detected
* Add a frontend interface for easier interaction
* Improve monitoring with real-time dashboards
* Optimize image model performance further

