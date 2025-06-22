import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from datasets import load_dataset
import joblib
import os
import numpy as np
import argparse
import re

# --- Data Loading and Normalization Functions ---

def parse_salary_range(salary_str):
    """Parses salary strings like '$76k-$97k' into min and max integers."""
    salary_str = salary_str.replace('$', '').replace('k', '000').replace('K','000')
    parts = [p.strip() for p in salary_str.split('-')]
    try:
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return None, None
    except ValueError:
        return None, None

def load_and_normalize_data():
    """Loads, normalizes, and combines multiple datasets."""
    all_dfs = []
    
    # --- Dataset 1: rishikeshv/job-descriptions-and-salary-dataset ---
    try:
        print("Loading dataset 1: rishikeshv/job-descriptions-and-salary-dataset")
        ds1 = load_dataset("rishikeshv/job-descriptions-and-salary-dataset", split="train")
        df1 = pd.DataFrame(ds1)
        df1 = df1[['job_title', 'job_location', 'job_description', 'min_salary', 'max_salary']]
        df1.columns = ['title', 'location', 'description', 'min_salary', 'max_salary']
        all_dfs.append(df1)
    except Exception as e:
        print(f"Could not load dataset 1. Error: {e}")

    # --- Dataset 2: ishaandey/job-positions-dataset ---
    try:
        print("Loading dataset 2: ishaandey/job-positions-dataset")
        ds2 = load_dataset("ishaandey/job-positions-dataset", split="train")
        df2 = pd.DataFrame(ds2)
        df2 = df2[['Job Title', 'Location', 'Job Description', 'Salary']]
        df2.columns = ['title', 'location', 'description', 'salary_str']
        
        salaries = df2['salary_str'].apply(parse_salary_range)
        df2['min_salary'], df2['max_salary'] = zip(*salaries)
        all_dfs.append(df2[['title', 'location', 'description', 'min_salary', 'max_salary']])
    except Exception as e:
        print(f"Could not load dataset 2. Error: {e}")
        
    if not all_dfs:
        raise RuntimeError("Could not load any datasets. Please check names and availability.")
        
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} records.")
    return combined_df

def main():
    parser = argparse.ArgumentParser(description="Train a salary prediction model on multiple datasets.")
    parser.add_argument("--output_dir", type=str, required=True, help="The directory to save the model.")
    args = parser.parse_args()

    MODEL_DIR = args.output_dir
    MODEL_FILE = os.path.join(MODEL_DIR, "salary_predictor_xgboost.json")
    PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "salary_predictor_preprocessor.joblib")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_normalize_data()

    print("Cleaning and preparing combined data...")
    df.dropna(subset=['min_salary', 'max_salary', 'description', 'location', 'title'], inplace=True)
    df['Salary.Avg'] = (df['min_salary'] + df['max_salary']) / 2
    
    top_locations = df['location'].value_counts().nlargest(20).index
    df['Location'] = df['location'].where(df['location'].isin(top_locations), 'Other')

    top_titles = df['title'].value_counts().nlargest(50).index
    df['Job.Title'] = df['title'].where(df['title'].isin(top_titles), 'Other')

    X = df[['Job.Title', 'Location', 'description']]
    y = df['Salary.Avg']

    preprocessor = ColumnTransformer(
        transformers=[
            ('title_cat', OneHotEncoder(handle_unknown='ignore'), ['Job.Title']),
            ('location_cat', OneHotEncoder(handle_unknown='ignore'), ['Location']),
            ('desc_tfidf', TfidfVectorizer(max_features=1000, stop_words='english', max_df=0.9, min_df=5)),
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, subsample=0.8, colsample_bytree=0.8))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model evaluation complete. RMSE on test data: ${rmse:,.2f}")

    print(f"Saving model and preprocessor to {MODEL_DIR}")
    model_pipeline.named_steps['regressor'].save_model(MODEL_FILE)
    joblib.dump(model_pipeline.named_steps['preprocessor'], PREPROCESSOR_FILE)
    print("Salary prediction model saved successfully.")

if __name__ == "__main__":
    main() 