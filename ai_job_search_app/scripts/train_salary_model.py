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
        
        # Map available columns to what we need
        column_mapping = {}
        
        # Look for title-like columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['title', 'position', 'job']):
                column_mapping['title'] = col
                break
        
        # Look for location-like columns (optional for this dataset)
        for col in df.columns:
            if 'location' in col.lower():
                column_mapping['location'] = col
                break
        
        # Look for description-like columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['description', 'summary', 'responsibilities']):
                column_mapping['description'] = col
                break
        
        # Look for salary columns
        for col in df.columns:
            if 'salary' in col.lower():
                column_mapping['salary'] = col
                break
        
        # Check if we have minimum required columns (title, description, salary)
        required_cols = ['title', 'description', 'salary']
        missing_cols = [col for col in required_cols if col not in column_mapping]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Use job_position as title if available
        if 'job_position' in df.columns:
            column_mapping['title'] = 'job_position'
        
        # Extract the columns we need
        cols_to_extract = [column_mapping[col] for col in ['title', 'description', 'salary']]
        if 'location' in column_mapping:
            cols_to_extract.append(column_mapping['location'])
        
        df = df[cols_to_extract].copy()
        df.dropna(subset=[column_mapping['salary']], inplace=True)

        def parse_salary(s):
            s = str(s).replace('$', '').replace(',', '').replace(' a year', '')
            parts = [p.strip() for p in s.split('-')]
            if len(parts) == 2:
                try:
                    return float(parts[0]), float(parts[1])
                except ValueError:
                    return None, None
            return None, None

        salaries = df[column_mapping['salary']].apply(parse_salary)
        df['min_salary'], df['max_salary'] = zip(*salaries)
        df.dropna(subset=['min_salary', 'max_salary'], inplace=True)
        
        # Create the rename mapping
        rename_mapping = {
            column_mapping['title']: 'title',
            column_mapping['description']: 'description'
        }
        if 'location' in column_mapping:
            rename_mapping[column_mapping['location']] = 'location'
        
        df.rename(columns=rename_mapping, inplace=True)
        
        # Return with or without location column
        if 'location' in df.columns:
            return df[['title', 'location', 'description', 'min_salary', 'max_salary']]
        else:
            # Add a dummy location column if missing
            df['location'] = 'Unknown'
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
        # Check available columns
        print(f"Azrai dataset columns: {df.columns.tolist()}")
        
        # Map columns dynamically
        column_mapping = {}
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['job_title', 'title']):
                column_mapping['title'] = col
            elif 'location' in col.lower():
                column_mapping['location'] = col
            elif any(keyword in col.lower() for keyword in ['job_description', 'description']):
                column_mapping['description'] = col
            elif 'salary' in col.lower():
                column_mapping['salary'] = col
        
        required_cols = ['title', 'description', 'salary']
        missing_cols = [col for col in required_cols if col not in column_mapping]
        
        if missing_cols:
            print(f"Azrai dataset missing columns: {missing_cols}")
            return pd.DataFrame()
        
        # Extract required columns
        cols_to_extract = [column_mapping[col] for col in required_cols]
        if 'location' in column_mapping:
            cols_to_extract.append(column_mapping['location'])
        
        df = df[cols_to_extract].copy()
        df.dropna(subset=[column_mapping['salary']], inplace=True)
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

        salaries = df[column_mapping['salary']].apply(parse_and_convert_salary)
        df['min_salary'], df['max_salary'] = zip(*salaries)
        df.dropna(subset=['min_salary', 'max_salary'], inplace=True)
        
        # Create rename mapping
        rename_mapping = {
            column_mapping['title']: 'title',
            column_mapping['description']: 'description'
        }
        if 'location' in column_mapping:
            rename_mapping[column_mapping['location']] = 'location'
        
        df.rename(columns=rename_mapping, inplace=True)
        
        # Return with or without location
        if 'location' in df.columns:
            return df[['title', 'location', 'description', 'min_salary', 'max_salary']]
        else:
            df['location'] = 'Unknown'
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

    # Ensure all text fields are strings and handle missing values
    combined_df['title'] = combined_df['title'].fillna('Unknown').astype(str)
    combined_df['location'] = combined_df['location'].fillna('Unknown').astype(str)
    combined_df['description'] = combined_df['description'].fillna('No description').astype(str)
    
    X = combined_df[['title', 'location', 'description']]
    y = combined_df['Salary.Avg']

    # Model Training Pipeline - use make_column_transformer for better handling
    from sklearn.compose import make_column_transformer
    
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', max_categories=100, sparse_output=False), ['title']),
        (OneHotEncoder(handle_unknown='ignore', max_categories=50, sparse_output=False), ['location']),
        (TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1,2)), 'description'),
        remainder='drop',
        sparse_threshold=0  # Force dense output
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