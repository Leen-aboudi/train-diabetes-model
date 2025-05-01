import pandas as pd                      
import numpy as np                       
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data and cross-validation
from sklearn.linear_model import LogisticRegression                    # Logistic Regression model
from sklearn.metrics import accuracy_score                             # To evaluate model performance
from sklearn.preprocessing import StandardScaler                       # To standardize feature values
import mlflow                                                           # For experiment tracking

# Configure MLflow experiment tracking
mlflow.set_tracking_uri('file:./mlruns')               # Set MLflow to log runs locally in ./mlruns folder
mlflow.set_experiment('diabetes_prediction')           # Create or use an experiment named 'diabetes_prediction'

# --------------------------
# Function to load the dataset
# --------------------------
def load_data():
    # Define column names for better readability
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    # Load the dataset directly from a URL (Pima Indians Diabetes Dataset)
    data = pd.read_csv(
        'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
        names=columns
    )
    return data

# --------------------------
# Function to clean and preprocess the data
# --------------------------
def preprocess_data(data):
    # Columns where zero values are invalid and should be treated as missing
    zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace all zero values in these columns with NaN
    data[zero_not_valid] = data[zero_not_valid].replace(0, np.nan)
    
    # Fill missing values (NaNs) with the median value of each respective column
    for column in zero_not_valid:
        data[column] = data[column].fillna(data[column].median())
    
    return data

def main():
    # Load and preprocess the dataset
    print("Loading and preprocessing data...")
    data = load_data()
    data = preprocess_data(data)
    
    # Separate input features (X) and target variable (y)
    X = data.drop('Outcome', axis=1)     # All columns except 'Outcome'
    y = data['Outcome']                  # Target variable: 0 = no diabetes, 1 = has diabetes
    
    # Standardize the feature values to have mean=0 and std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42   # random_state ensures reproducibility
    )
    
    # Start tracking a new MLflow run
    with mlflow.start_run():
        # Initialize the Logistic Regression model
        model = LogisticRegression(
            random_state=42,    # Ensure consistent behavior across runs
            max_iter=500,       # Increase max iterations for better convergence
            tol=1e-4            # Convergence tolerance (when to stop training)
        )
        
        # Perform 5-fold cross-validation to validate model performance
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        
        # Log model parameters to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)  # Even though model uses 500, logging 1000 might be a mistake
        mlflow.log_param("tol", 1e-4)
        
        # Train the model on the training data
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Predict outcomes on the test data
        y_pred = model.predict(X_test)
        
        # Calculate and log performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
        
        # Log the trained model to MLflow with input/output signature
        signature = mlflow.models.signature.infer_signature(
            X_train, model.predict(X_train)
        )
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=X_train[:5]  # Example input for reference
        )
        
        # Display results to the console
        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nModel training completed! Check MLflow UI for detailed results.")
        print("Run 'mlflow ui' and open http://127.0.0.1:5000 in your browser.")

if __name__ == "__main__":
    main()
