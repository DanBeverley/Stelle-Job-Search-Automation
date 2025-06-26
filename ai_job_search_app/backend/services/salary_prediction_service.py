import joblib
import xgboost as xgb
import pandas as pd
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SalaryPredictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SalaryPredictor, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.preprocessor = None
            cls._instance.model_loaded = False
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        if self.model_loaded:
            return
        
        # Updated paths to point to final_model directory
        MODEL_DIR = "ai_job_search_app/final_model"
        MODEL_FILE = os.path.join(MODEL_DIR, "salary_predictor_xgboost.json")
        PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "salary_predictor_preprocessor.joblib")

        try:
            if not os.path.exists(MODEL_FILE) or not os.path.exists(PREPROCESSOR_FILE):
                raise FileNotFoundError("Salary model or preprocessor not found.")

            logger.info("Loading salary prediction model and preprocessor...")
            self.model = xgb.XGBRegressor()
            self.model.load_model(MODEL_FILE)
            self.preprocessor = joblib.load(PREPROCESSOR_FILE)
            self.model_loaded = True
            logger.info("Salary prediction model loaded successfully")
        except Exception as e:
            logger.error("Error loading salary model: %s", str(e))
            self.model_loaded = False

    def predict(self, job_title: str, location: str, job_description: str) -> float:
        if not self.model_loaded:
            raise RuntimeError("Salary prediction model is not loaded.")
        
        # Create a DataFrame from the input (using column names expected by the model)
        input_data = pd.DataFrame({
            'title': [job_title],
            'location': [location],
            'description': [job_description]
        })

        # The preprocessor and model expect a DataFrame
        prediction = self.model.predict(self.preprocessor.transform(input_data))
        return float(prediction[0])

# --- Service Function ---
salary_predictor = SalaryPredictor()

def predict_salary_with_xgb(job_title: str, location: str, job_description: str) -> Dict[str, Any]:
    """
    Predicts salary using the fine-tuned XGBoost model.
    """
    if not salary_predictor.model_loaded:
        raise RuntimeError("Salary prediction model is not available.")
    
    predicted_avg = salary_predictor.predict(job_title, location, job_description)

    # For now, we'll create a simple range around the predicted average
    # A more advanced model could predict min and max directly
    min_salary = int(predicted_avg * 0.9)
    max_salary = int(predicted_avg * 1.1)

    return {
        "min_salary": min_salary,
        "max_salary": max_salary,
        "median_salary": int(predicted_avg),
        "commentary": "Salary prediction based on our custom model considering job title, description, and location.",
        "currency": "USD", # Assuming USD from the dataset
        "period": "annual"
    } 