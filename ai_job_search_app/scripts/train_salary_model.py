import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from datasets import load_dataset, DownloadConfig
import joblib
import os
import numpy as np
import argparse
import re
import time
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_dataset_with_retry(dataset_name, max_retries=3, timeout=60, **kwargs):
    """Load dataset with retry logic and longer timeout"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to load {dataset_name}")
            
            download_config = DownloadConfig(
                max_retries=3,
                num_proc=1,
                resume_download=True
            )
            
            dataset = load_dataset(
                dataset_name, 
                download_config=download_config,
                **kwargs
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                logger.info(f"Waiting {wait_time} seconds before retry")
                time.sleep(wait_time)
            else:
                raise e

def create_synthetic_data():
    """Create synthetic salary data as fallback"""
    logger.info("Creating synthetic salary data as fallback")
    
    np.random.seed(42)
    n_samples = 5000
    
    titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer', 
              'Frontend Developer', 'Backend Developer', 'ML Engineer', 'Data Analyst',
              'Project Manager', 'QA Engineer', 'Full Stack Developer', 'UI/UX Designer']
    
    locations = ['San Francisco', 'New York', 'Seattle', 'Austin', 'Boston', 'Chicago',
                 'Los Angeles', 'Denver', 'Portland', 'Remote', 'Washington DC', 'Atlanta']
    
    title_base_salaries = {
        'Software Engineer': 120000, 'Data Scientist': 130000, 'Product Manager': 140000,
        'DevOps Engineer': 125000, 'Frontend Developer': 110000, 'Backend Developer': 115000,
        'ML Engineer': 145000, 'Data Analyst': 95000, 'Project Manager': 115000,
        'QA Engineer': 90000, 'Full Stack Developer': 115000, 'UI/UX Designer': 105000
    }
    
    location_multipliers = {
        'San Francisco': 1.4, 'New York': 1.3, 'Seattle': 1.2, 'Austin': 1.0,
        'Boston': 1.2, 'Chicago': 1.1, 'Los Angeles': 1.2, 'Denver': 1.1,
        'Portland': 1.1, 'Remote': 0.95, 'Washington DC': 1.2, 'Atlanta': 1.0
    }
    
    data = []
    for _ in range(n_samples):
        title = np.random.choice(titles)
        location = np.random.choice(locations)
        
        base_salary = title_base_salaries[title]
        location_mult = location_multipliers[location]
        
        salary_variation = np.random.uniform(0.8, 1.2)
        avg_salary = base_salary * location_mult * salary_variation
        
        min_salary = avg_salary * 0.9
        max_salary = avg_salary * 1.1
        
        skills = np.random.choice(['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'Docker', 
                                  'React', 'Node.js', 'Kubernetes', 'Machine Learning'], 
                                 size=np.random.randint(3, 7), replace=False)
        
        description = f"Looking for a {title} with experience in {', '.join(skills)}. " \
                     f"Great opportunity for growth and learning. Competitive salary and benefits."
        
        data.append({
            'title': title,
            'location': location,
            'description': description,
            'min_salary': min_salary,
            'max_salary': max_salary
        })
    
    logger.info(f"Generated {len(data)} synthetic salary records")
    return pd.DataFrame(data)

def normalize_linkedin_data():
    """Loads and normalizes the LinkedIn dataset."""
    try:
        logger.info("Loading dataset 1: xanderios/linkedin-job-postings")
        dataset = load_dataset_with_retry(
            "xanderios/linkedin-job-postings", 
            data_files="job_postings.csv", 
            split="train"
        )
        df = pd.DataFrame(dataset)
        df = df.filter(['title', 'location', 'description', 'med_salary', 'pay_period'])
        df.dropna(subset=['med_salary', 'pay_period'], inplace=True)

        def normalize_salary(row):
            if row['pay_period'] == 'YEARLY':
                return row['med_salary']
            if row['pay_period'] == 'MONTHLY':
                return row['med_salary'] * 12
            if row['pay_period'] == 'HOURLY':
                return row['med_salary'] * 40 * 52
            return None
        
        df['yearly_salary'] = df.apply(normalize_salary, axis=1)
        df.dropna(subset=['yearly_salary'], inplace=True)
        df['min_salary'] = df['yearly_salary'] * 0.85
        df['max_salary'] = df['yearly_salary'] * 1.15
        logger.info(f"Successfully processed LinkedIn dataset with {len(df)} records")
        return df[['title', 'location', 'description', 'min_salary', 'max_salary']]
    except Exception as e:
        logger.error(f"Could not process LinkedIn dataset: {e}")
        return pd.DataFrame()

def normalize_classification_data():
    """Loads and normalizes the job posting classification dataset."""
    try:
        logger.info("Loading dataset 2: will4381/job-posting-classification")
        dataset = load_dataset_with_retry("will4381/job-posting-classification", split="train")
        df = pd.DataFrame(dataset)
        
        logger.debug(f"Available columns: {df.columns.tolist()}")
        
        column_mapping = {}
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['title', 'position', 'job']):
                column_mapping['title'] = col
                break
        
        for col in df.columns:
            if 'location' in col.lower():
                column_mapping['location'] = col
                break
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['description', 'summary', 'responsibilities']):
                column_mapping['description'] = col
                break
        
        for col in df.columns:
            if 'salary' in col.lower():
                column_mapping['salary'] = col
                break
        
        required_cols = ['title', 'description', 'salary']
        missing_cols = [col for col in required_cols if col not in column_mapping]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            logger.warning(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        if 'job_position' in df.columns:
            column_mapping['title'] = 'job_position'
        
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
        
        rename_mapping = {
            column_mapping['title']: 'title',
            column_mapping['description']: 'description'
        }
        if 'location' in column_mapping:
            rename_mapping[column_mapping['location']] = 'location'
        
        df.rename(columns=rename_mapping, inplace=True)
        
        if 'location' in df.columns:
            result_df = df[['title', 'location', 'description', 'min_salary', 'max_salary']]
        else:
            df['location'] = 'Unknown'
            result_df = df[['title', 'location', 'description', 'min_salary', 'max_salary']]
        
        logger.info(f"Successfully processed classification dataset with {len(result_df)} records")
        return result_df
    except Exception as e:
        logger.error(f"Could not process classification dataset: {e}")
        return pd.DataFrame()

def normalize_azrai_data():
    """Loads, normalizes, and converts currency for the Azrai dataset."""
    try:
        logger.info("Loading dataset 3: azrai99/job-dataset")
        dataset = load_dataset_with_retry("azrai99/job-dataset", split="train")
        df = pd.DataFrame(dataset)
        logger.debug(f"Azrai dataset columns: {df.columns.tolist()}")
        
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
            logger.warning(f"Azrai dataset missing columns: {missing_cols}")
            return pd.DataFrame()
        
        cols_to_extract = [column_mapping[col] for col in required_cols]
        if 'location' in column_mapping:
            cols_to_extract.append(column_mapping['location'])
        
        df = df[cols_to_extract].copy()
        df.dropna(subset=[column_mapping['salary']], inplace=True)
        
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
        
        rename_mapping = {
            column_mapping['title']: 'title',
            column_mapping['description']: 'description'
        }
        if 'location' in column_mapping:
            rename_mapping[column_mapping['location']] = 'location'
        
        df.rename(columns=rename_mapping, inplace=True)
        
        if 'location' in df.columns:
            result_df = df[['title', 'location', 'description', 'min_salary', 'max_salary']]
        else:
            df['location'] = 'Unknown'
            result_df = df[['title', 'location', 'description', 'min_salary', 'max_salary']]
        
        logger.info(f"Successfully processed Azrai dataset with {len(result_df)} records")
        return result_df
    except Exception as e:
        logger.error(f"Could not process Azrai dataset: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Train a salary prediction model using multiple datasets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model artifacts.")
    args = parser.parse_args()

    logger.info("Starting salary model training pipeline")
    
    os.makedirs(args.output_dir, exist_ok=True)
    MODEL_FILE = os.path.join(args.output_dir, "salary_predictor_xgboost.json")
    PREPROCESSOR_FILE = os.path.join(args.output_dir, "salary_predictor_preprocessor.joblib")

    logger.info("Loading and processing datasets")
    df1 = normalize_linkedin_data()
    df2 = normalize_classification_data()
    df3 = normalize_azrai_data()
    
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    if combined_df.empty:
        logger.warning("All external datasets failed to load. Using synthetic data instead")
        combined_df = create_synthetic_data()
    
    logger.info(f"Combined dataset has {len(combined_df)} records after normalization")

    logger.info("Preparing final features for training")
    combined_df.dropna(inplace=True)
    combined_df['Salary.Avg'] = (combined_df['min_salary'] + combined_df['max_salary']) / 2

    q_low = combined_df['Salary.Avg'].quantile(0.01)
    q_hi  = combined_df['Salary.Avg'].quantile(0.99)
    combined_df = combined_df[(combined_df['Salary.Avg'] > q_low) & (combined_df['Salary.Avg'] < q_hi)]
    logger.info(f"Dataset has {len(combined_df)} records after removing outliers")

    combined_df['title'] = combined_df['title'].fillna('Unknown').astype(str)
    combined_df['location'] = combined_df['location'].fillna('Unknown').astype(str)
    combined_df['description'] = combined_df['description'].fillna('No description').astype(str)

    X = combined_df[['title', 'location', 'description']]
    y = combined_df['Salary.Avg']

    from sklearn.compose import make_column_transformer
    
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', max_categories=100, sparse_output=False), ['title']),
        (OneHotEncoder(handle_unknown='ignore', max_categories=50, sparse_output=False), ['location']),
        (TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1,2)), 'description'),
        remainder='drop',
        sparse_threshold=0
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
    
    logger.info("Training XGBoost model")
    model_pipeline.fit(X_train, y_train)
    
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info(f"Model training complete. RMSE on test data: ${rmse:,.2f}")

    logger.info(f"Saving model and preprocessor to {args.output_dir}")
    model_pipeline.named_steps['regressor'].save_model(MODEL_FILE)
    joblib.dump(model_pipeline.named_steps['preprocessor'], PREPROCESSOR_FILE)
    logger.info("Model artifacts saved successfully")

if __name__ == "__main__":
    main() 