import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
import joblib
import os
import numpy as np

def main():
    # --- 1. Configuration ---
    DATASET_NAME = "aravind-g/job-salary-prediction"
    MODEL_DIR = "ai_job_search_app/data/models/salary_predictor"
    MODEL_FILE = os.path.join(MODEL_DIR, "xgb_salary_model.json")
    PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "preprocessor.joblib")

    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 2. Load Dataset ---
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(dataset)

    # --- 3. Initial Data Cleaning and Preparation ---
    print("Cleaning and preparing data...")
    # Drop rows with missing salary or crucial features
    df.dropna(subset=['Salary.Min', 'Salary.Max', 'Job.Description', 'Location'], inplace=True)
    
    # Create a single salary target by averaging Min and Max
    df['Salary.Avg'] = (df['Salary.Min'] + df['Salary.Max']) / 2
    # For simplicity, we'll predict the average. We could also predict a range.
    
    # Handle categorical features with many unique values by taking the top N
    top_locations = df['Location'].value_counts().nlargest(20).index
    df['Location'] = df['Location'].where(df['Location'].isin(top_locations), 'Other')

    top_titles = df['Job.Title'].value_counts().nlargest(50).index
    df['Job.Title'] = df['Job.Title'].where(df['Job.Title'].isin(top_titles), 'Other')

    # Define features and target
    X = df[['Job.Title', 'Location', 'Job.Description']]
    y = df['Salary.Avg']

    # --- 4. Define Preprocessing Steps ---
    print("Defining preprocessing pipeline...")
    # We create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('title_cat', OneHotEncoder(handle_unknown='ignore'), ['Job.Title']),
            ('location_cat', OneHotEncoder(handle_unknown='ignore'), ['Location']),
            ('desc_tfidf', TfidfVectorizer(max_features=500, stop_words='english'), 'Job.Description')
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # --- 5. Create the Full Model Pipeline ---
    # The pipeline will first preprocess the data, then train the XGBoost model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,          # Can be tuned with cross-validation
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1                  # Use all available CPU cores
        ))
    ])

    # --- 6. Split Data and Train the Model ---
    print("Splitting data and training the model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 7. Evaluate the Model ---
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model evaluation complete. Root Mean Squared Error: ${rmse:,.2f}")

    # --- 8. Save the Model and Preprocessor ---
    print(f"Saving model to {MODEL_FILE}")
    # The XGBoost model itself is saved separately for clarity
    model_pipeline.named_steps['regressor'].save_model(MODEL_FILE)

    print(f"Saving preprocessor to {PREPROCESSOR_FILE}")
    # We save the preprocessor from the pipeline
    joblib.dump(model_pipeline.named_steps['preprocessor'], PREPROCESSOR_FILE)
    
    print("Salary prediction model and preprocessor saved successfully.")

if __name__ == "__main__":
    main() 