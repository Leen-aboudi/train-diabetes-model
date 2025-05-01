# Diabetes Prediction Model with MLflow Tracking

This project implements a machine learning model to predict diabetes using the Pima Indians Diabetes dataset. The implementation uses Logistic Regression and MLflow for experiment tracking.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── train_diabetes_model.py
└── mlruns/
└── gitignore/
```

## Setup Instructions

1. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Model

1. Execute the training script:
   ```bash
   python3 train_diabetes_model.py
   ```

2. View MLflow tracking results:
   ```bash
   mlflow ui
   ```
   Then open http://127.0.0.1:5000 in your web browser

## Features

- Data preprocessing and handling of missing values
- Model training using Logistic Regression
- Performance evaluation using accuracy metric
- Experiment tracking with MLflow
- Model artifact storage

## Dataset

The Pima Indians Diabetes dataset includes the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target variable)

## MLflow Tracking

The following metrics and parameters are tracked:
- Model type
- Accuracy score
- Trained model artifact
- Cross-validation accuracy

Results can be viewed through the MLflow UI.