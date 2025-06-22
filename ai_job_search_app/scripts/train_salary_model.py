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
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a salary prediction model.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ai_job_search_app/data/models/salary_predictor",
        help="The directory to save the trained model and preprocessor."
    )
    args = parser.parse_args()

    # --- 1. Configuration ---
    DATASET_NAME = "sahirmaharaj/job-postings-from-google"
    MODEL_DIR = args.output_dir
    MODEL_FILE = os.path.join(MODEL_DIR, "salary_predictor_xgboost.json")
    PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "salary_predictor_preprocessor.joblib")

    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 2. Load Dataset ---
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(dataset)

    # --- 3. Initial Data Cleaning and Preparation ---
    print("Cleaning and preparing data...")
    # Adjusting column names for the new dataset
    df.dropna(subset=['salary_min', 'salary_max', 'description', 'location'], inplace=True)
    
    df['Salary.Avg'] = (df['salary_min'] + df['salary_max']) / 2
    
    top_locations = df['location'].value_counts().nlargest(20).index
    df['Location'] = df['location'].where(df['location'].isin(top_locations), 'Other')

    top_titles = df['title'].value_counts().nlargest(50).index
    df['Job.Title'] = df['title'].where(df['title'].isin(top_titles), 'Other')

    # Define features and target using new column names
    X = df[['Job.Title', 'Location', 'description']]
    y = df['Salary.Avg']

    # --- 4. Define Preprocessing Steps ---
    print("Defining preprocessing pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('title_cat', OneHotEncoder(handle_unknown='ignore'), ['Job.Title']),
            ('location_cat', OneHotEncoder(handle_unknown='ignore'), ['Location']),
            ('desc_tfidf', TfidfVectorizer(max_features=500, stop_words='english'), 'description')
        ],
        remainder='passthrough'
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