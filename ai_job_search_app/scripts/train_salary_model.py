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

def normalize_linkedin_data():
    """Loads and normalizes the LinkedIn dataset."""
    try:
        print("Loading dataset 1: xanderios/linkedin-job-postings")
        dataset = load_dataset("xanderios/linkedin-job-postings", data_files="job_postings.csv", split="train")
        df = pd.DataFrame(dataset)
        df = df.filter(['title', 'location', 'description', 'med_salary', 'pay_period'])
        df.dropna(subset=['med_salary', 'pay_period'], inplace=True)

        def normalize_salary(row):
            if row['pay_period'] == 'YEARLY':
                return row['med_salary']
            if row['pay_period'] == 'MONTHLY':
                return row['med_salary'] * 12
            if row['pay_period'] == 'HOURLY':
                return row['med_salary'] * 40 * 52 # 40 hours/week, 52 weeks/year
            return None
        
        df['yearly_salary'] = df.apply(normalize_salary, axis=1)
        df.dropna(subset=['yearly_salary'], inplace=True)
        # Create a plausible range from the median salary
        df['min_salary'] = df['yearly_salary'] * 0.85
        df['max_salary'] = df['yearly_salary'] * 1.15
        return df[['title', 'location', 'description', 'min_salary', 'max_salary']]
    except Exception as e:
        print(f"Could not process LinkedIn dataset. Error: {e}")
        return pd.DataFrame()

def normalize_classification_data():
    """Loads and normalizes the job posting classification dataset."""
    try:
        print("Loading dataset 2: will4381/job-posting-classification")
        dataset = load_dataset("will4381/job-posting-classification", split="train")
        df = pd.DataFrame(dataset)
        
        # Print available columns for debugging
        print(f"Available columns: {df.columns.tolist()}")
        
        # Try different possible column names
        title_col = None
        location_col = None
        description_col = None
        salary_col = None
        
        for col in df.columns:
            if 'title' in col.lower():
                title_col = col
            elif 'location' in col.lower():
                location_col = col
            elif 'description' in col.lower():
                description_col = col
            elif 'salary' in col.lower():
                salary_col = col
        
        if not all([title_col, location_col, description_col, salary_col]):
            print(f"Missing required columns. Found: title={title_col}, location={location_col}, description={description_col}, salary={salary_col}")
            return pd.DataFrame()
        
        df = df[[title_col, location_col, description_col, salary_col]].copy()
        df.dropna(subset=[salary_col], inplace=True)

        def parse_salary(s):
            s = str(s).replace('$', '').replace(',', '').replace(' a year', '')
            parts = [p.strip() for p in s.split('-')]
            if len(parts) == 2:
                try:
                    return float(parts[0]), float(parts[1])
                except ValueError:
                    return None, None
            return None, None

        salaries = df[salary_col].apply(parse_salary)
        df['min_salary'], df['max_salary'] = zip(*salaries)
        df.dropna(subset=['min_salary', 'max_salary'], inplace=True)
        df.rename(columns={title_col: 'title', location_col: 'location', description_col: 'description'}, inplace=True)
        return df[['title', 'location', 'description', 'min_salary', 'max_salary']]
    except Exception as e:
        print(f"Could not process classification dataset. Error: {e}")
        return pd.DataFrame()

def normalize_azrai_data():
    """Loads, normalizes, and converts currency for the Azrai dataset."""
    try:
        print("Loading dataset 3: azrai99/job-dataset")
        dataset = load_dataset("azrai99/job-dataset", split="train")
        df = pd.DataFrame(dataset)
        df = df.filter(['job_title', 'location', 'job_description', 'salary'])
        df.dropna(subset=['salary'], inplace=True)
        # Currency conversion rate (approximate)
        RM_TO_USD = 1 / 4.7 

        def parse_and_convert_salary(s):
            s = str(s).lower().replace(',', '').strip()
            s = re.sub(r'\.+$', '', s) 
            numbers = []
            for n in re.findall(r'\d+(?:\.\d+)?', s):  
                if n and n != '.' and n != '':
                    try:
                        numbers.append(float(n))
                    except ValueError:
                        continue
            
            if len(numbers) < 2: return None, None
            
            min_sal, max_sal = numbers[0], numbers[1]
            if 'rm' in s:
                min_sal *= RM_TO_USD
                max_sal *= RM_TO_USD
            if 'month' in s:
                min_sal *= 12
                max_sal *= 12

            return min_sal, max_sal

        salaries = df['salary'].apply(parse_and_convert_salary)
        df['min_salary'], df['max_salary'] = zip(*salaries)
        df.dropna(subset=['min_salary', 'max_salary'], inplace=True)
        df.rename(columns={'job_title': 'title', 'job_description': 'description'}, inplace=True)
        return df[['title', 'location', 'description', 'min_salary', 'max_salary']]
    except Exception as e:
        print(f"Could not process Azrai dataset. Error: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Train a salary prediction model using multiple datasets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model artifacts.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    MODEL_FILE = os.path.join(args.output_dir, "salary_predictor_xgboost.json")
    PREPROCESSOR_FILE = os.path.join(args.output_dir, "salary_predictor_preprocessor.joblib")

    # Load, Normalize, and Combine Data 
    df1 = normalize_linkedin_data()
    df2 = normalize_classification_data()
    df3 = normalize_azrai_data()
    
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    if combined_df.empty:
        raise RuntimeError("All datasets failed to load. Aborting training.")
    print(f"\n--- Combined dataset has {len(combined_df)} records after normalization. ---")

    # Feature Engineering
    print("Preparing final features for training...")
    combined_df.dropna(inplace=True)
    combined_df['Salary.Avg'] = (combined_df['min_salary'] + combined_df['max_salary']) / 2

    # Remove outliers for more stable training
    q_low = combined_df['Salary.Avg'].quantile(0.01)
    q_hi  = combined_df['Salary.Avg'].quantile(0.99)
    combined_df = combined_df[(combined_df['Salary.Avg'] > q_low) & (combined_df['Salary.Avg'] < q_hi)]
    print(f"Dataset has {len(combined_df)} records after removing outliers.")

    X = combined_df[['title', 'location', 'description']]
    y = combined_df['Salary.Avg']

    # Model Training Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('title', OneHotEncoder(handle_unknown='ignore', max_categories=100), ['title']),
            ('location', OneHotEncoder(handle_unknown='ignore', max_categories=50), ['location']),
            ('description', TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1,2)), ['description']),
        ],
        remainder='drop'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, 
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Training XGBoost model... ---")
    # Fit the model without early stopping for simplicity
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"--- Model training complete. RMSE on test data: ${rmse:,.2f} ---")

    # Save Artifacts 
    print(f"\nSaving model and preprocessor to {args.output_dir}")
    model_pipeline.named_steps['regressor'].save_model(MODEL_FILE)
    joblib.dump(model_pipeline.named_steps['preprocessor'], PREPROCESSOR_FILE)
    print("--- Artifacts saved successfully. ---")

if __name__ == "__main__":
    main() 