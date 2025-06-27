#!/usr/bin/env python3
"""
Salary Prediction Model Training Script using Transformers (consistent with other models)
This creates a consistent LoRA adapter format like cover_letter and interview models
"""
import torch
import pandas as pd
import numpy as np
import os
import argparse
import logging
import time
import re
from datasets import Dataset, load_dataset, DownloadConfig
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_synthetic_salary_data(n_samples=8000):
    """Create comprehensive synthetic salary data"""
    set_random_seeds(42)
    
    # Expanded job titles with base salaries
    job_data = {
        'Software Engineer': 125000, 'Senior Software Engineer': 155000, 'Staff Software Engineer': 185000,
        'Data Scientist': 135000, 'Senior Data Scientist': 165000, 'Principal Data Scientist': 195000,
        'Product Manager': 140000, 'Senior Product Manager': 170000, 'Principal Product Manager': 200000,
        'DevOps Engineer': 130000, 'Senior DevOps Engineer': 160000, 'Platform Engineer': 175000,
        'Frontend Developer': 115000, 'Senior Frontend Developer': 145000, 'Lead Frontend Developer': 165000,
        'Backend Developer': 120000, 'Senior Backend Developer': 150000, 'Lead Backend Developer': 170000,
        'Full Stack Developer': 118000, 'Senior Full Stack Developer': 148000, 'Lead Full Stack Developer': 168000,
        'Machine Learning Engineer': 150000, 'Senior ML Engineer': 180000, 'Principal ML Engineer': 210000,
        'Data Engineer': 125000, 'Senior Data Engineer': 155000, 'Principal Data Engineer': 185000,
        'Security Engineer': 140000, 'Senior Security Engineer': 170000, 'Principal Security Engineer': 200000,
        'QA Engineer': 95000, 'Senior QA Engineer': 125000, 'QA Lead': 145000,
        'Technical Lead': 160000, 'Engineering Manager': 175000, 'Senior Engineering Manager': 205000,
        'UI/UX Designer': 105000, 'Senior UI/UX Designer': 135000, 'Design Lead': 155000,
        'Cloud Engineer': 135000, 'Senior Cloud Engineer': 165000, 'Cloud Architect': 185000,
        'Site Reliability Engineer': 145000, 'Senior SRE': 175000, 'Principal SRE': 205000,
        'Cybersecurity Analyst': 110000, 'Senior Cybersecurity Analyst': 140000, 'Security Architect': 170000,
        'Database Administrator': 105000, 'Senior DBA': 135000, 'Database Architect': 165000,
        'Business Analyst': 85000, 'Senior Business Analyst': 115000, 'Principal Business Analyst': 145000,
        'Project Manager': 115000, 'Senior Project Manager': 145000, 'Program Manager': 165000,
        'Scrum Master': 105000, 'Senior Scrum Master': 135000, 'Agile Coach': 155000
    }
    
    # Location multipliers based on real market data
    location_data = {
        'San Francisco, CA': 1.45, 'New York, NY': 1.35, 'Seattle, WA': 1.25, 'Los Angeles, CA': 1.20,
        'Boston, MA': 1.22, 'Washington, DC': 1.20, 'Chicago, IL': 1.15, 'Austin, TX': 1.05,
        'Denver, CO': 1.10, 'Portland, OR': 1.12, 'Atlanta, GA': 1.05, 'Raleigh, NC': 1.00,
        'Phoenix, AZ': 1.02, 'Dallas, TX': 1.08, 'Miami, FL': 1.10, 'Minneapolis, MN': 1.08,
        'Philadelphia, PA': 1.12, 'San Diego, CA': 1.18, 'Nashville, TN': 1.00, 'Remote': 0.98,
        'Oakland, CA': 1.40, 'Mountain View, CA': 1.50, 'Palo Alto, CA': 1.48, 'Cupertino, CA': 1.46,
        'Redmond, WA': 1.28, 'Bellevue, WA': 1.26, 'Cambridge, MA': 1.25, 'Santa Clara, CA': 1.42,
        'Sunnyvale, CA': 1.44, 'San Jose, CA': 1.38, 'Menlo Park, CA': 1.47, 'Fremont, CA': 1.35
    }
    
    # Experience multipliers
    experience_multipliers = {
        '0-1 years': 0.75, '1-2 years': 0.85, '2-3 years': 0.95, '3-5 years': 1.0,
        '5-7 years': 1.15, '7-10 years': 1.30, '10+ years': 1.45
    }
    
    # Company size multipliers
    company_multipliers = {
        'Startup (1-50)': 0.90, 'Small (51-200)': 0.95, 'Medium (201-1000)': 1.0,
        'Large (1001-5000)': 1.08, 'Enterprise (5000+)': 1.15, 'FAANG': 1.25
    }
    
    # Skill categories that affect salary
    high_value_skills = ['Machine Learning', 'AI', 'Deep Learning', 'Kubernetes', 'AWS', 'Azure', 'GCP']
    standard_skills = ['Python', 'Java', 'JavaScript', 'React', 'Node.js', 'SQL', 'Docker', 'Git']
    
    data = []
    
    for _ in range(n_samples):
        # Random selections
        title = np.random.choice(list(job_data.keys()))
        location = np.random.choice(list(location_data.keys()))
        experience = np.random.choice(list(experience_multipliers.keys()))
        company_size = np.random.choice(list(company_multipliers.keys()))
        
        # Base salary calculation
        base_salary = job_data[title]
        location_mult = location_data[location]
        exp_mult = experience_multipliers[experience]
        company_mult = company_multipliers[company_size]
        
        # Skill bonus
        num_high_value = np.random.randint(0, 4)
        num_standard = np.random.randint(2, 6)
        skills = (np.random.choice(high_value_skills, size=num_high_value, replace=False).tolist() +
                 np.random.choice(standard_skills, size=num_standard, replace=False).tolist())
        
        skill_bonus = 1.0 + (num_high_value * 0.05)  # 5% bonus per high-value skill
        
        # Random variation
        variation = np.random.uniform(0.90, 1.10)
        
        # Final salary calculation
        final_salary = base_salary * location_mult * exp_mult * company_mult * skill_bonus * variation
        
        # Create min/max range
        min_salary = final_salary * 0.90
        max_salary = final_salary * 1.10
        
        # Create job description
        description = f"We are seeking a {title} with {experience} of experience. " \
                     f"Must have expertise in {', '.join(skills[:3])}. " \
                     f"Based in {location}. {company_size} company offering competitive compensation."
        
        data.append({
            'title': title,
            'location': location,
            'experience': experience,
            'company_size': company_size,
            'skills': ', '.join(skills),
            'description': description,
            'min_salary': min_salary,
            'max_salary': max_salary,
            'avg_salary': (min_salary + max_salary) / 2
        })
    
    logger.info(f"Generated {len(data)} synthetic salary records")
    return pd.DataFrame(data)

def load_real_salary_data():
    """Attempt to load real salary datasets with fallback to synthetic"""
    datasets_to_try = [
        "xanderios/linkedin-job-postings",
        "jacob-hugging-face/job-descriptions",
        "lukebarousse/data_jobs"
    ]
    
    for dataset_name in datasets_to_try:
        try:
            logger.info(f"Attempting to load dataset: {dataset_name}")
            
            download_config = DownloadConfig(
                max_retries=2,
                num_proc=1,
                resume_download=True
            )
            
            dataset = load_dataset(dataset_name, download_config=download_config, split="train")
            df = pd.DataFrame(dataset)
            
            # Try to identify relevant columns
            potential_salary_cols = ['salary', 'med_salary', 'avg_salary', 'compensation', 'pay']
            potential_title_cols = ['title', 'job_title', 'position', 'role']
            potential_location_cols = ['location', 'city', 'state', 'country']
            potential_desc_cols = ['description', 'job_description', 'details', 'summary']
            
            salary_col = None
            title_col = None
            location_col = None
            desc_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if any(sal_col in col_lower for sal_col in potential_salary_cols) and salary_col is None:
                    salary_col = col
                if any(title_col in col_lower for title_col in potential_title_cols) and title_col is None:
                    title_col = col
                if any(loc_col in col_lower for loc_col in potential_location_cols) and location_col is None:
                    location_col = col
                if any(desc_col in col_lower for desc_col in potential_desc_cols) and desc_col is None:
                    desc_col = col
            
            if salary_col and title_col:
                logger.info(f"Found viable dataset with salary column: {salary_col}")
                
                # Basic data cleaning
                df = df.dropna(subset=[salary_col, title_col])
                
                # Normalize salary column
                df['normalized_salary'] = pd.to_numeric(df[salary_col], errors='coerce')
                df = df.dropna(subset=['normalized_salary'])
                
                # Filter realistic salary ranges (30k to 500k)
                df = df[(df['normalized_salary'] >= 30000) & (df['normalized_salary'] <= 500000)]
                
                if len(df) > 100:  # Need at least 100 examples
                    logger.info(f"Successfully loaded {len(df)} real salary records from {dataset_name}")
                    return df, dataset_name
                    
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            continue
    
    logger.warning("Could not load any real datasets, using synthetic data")
    return None, None

def prepare_salary_text_data(df, is_real_data=False, dataset_name=None):
    """Convert salary data to text format for language model training"""
    
    processed_data = []
    
    for _, row in df.iterrows():
        try:
            if is_real_data:
                # Extract available information from real dataset
                title = str(row.get('title', row.get('job_title', 'Software Engineer')))
                location = str(row.get('location', row.get('city', 'Remote')))
                
                # Try to extract salary
                salary = None
                for col in ['normalized_salary', 'med_salary', 'salary', 'avg_salary']:
                    if col in row and pd.notna(row[col]):
                        salary = float(row[col])
                        break
                
                if not salary:
                    continue
                
                # Create min/max from single salary value
                min_salary = salary * 0.9
                max_salary = salary * 1.1
                
                description = str(row.get('description', row.get('job_description', 
                    f"Position for {title} in {location}. Competitive salary and benefits.")))[:500]
                
            else:
                # Use synthetic data structure
                title = row['title']
                location = row['location']
                min_salary = row['min_salary']
                max_salary = row['max_salary']
                description = row['description'][:500]
            
            # Create standardized prompt format
            prompt = f"""Job Analysis Request:
Title: {title}
Location: {location}
Description: {description}

Please provide salary prediction:"""

            response = f"""Based on the job details provided:

Position: {title}
Location: {location}

Salary Analysis:
- Minimum Salary: ${min_salary:,.0f}
- Maximum Salary: ${max_salary:,.0f}
- Average Salary: ${(min_salary + max_salary) / 2:,.0f}

This salary range is based on current market conditions, location, and role requirements."""

            # Format for instruction tuning
            text = f"""### Human: {prompt}

### Assistant: {response}

### End"""

            processed_data.append({"text": text})
            
        except Exception as e:
            logger.warning(f"Error processing row: {e}")
            continue
    
    logger.info(f"Processed {len(processed_data)} salary examples for training")
    return processed_data

class SalaryLossCallback(TrainerCallback):
    """Custom callback for salary model training"""
    def __init__(self, target_loss=1.0):
        self.target_loss = target_loss
        self.best_loss = float('inf')
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            current_loss = logs['eval_loss']
            logger.info(f"Eval loss: {current_loss:.4f} | Target: <{self.target_loss}")
            
            if current_loss < self.target_loss:
                logger.info(f"Target achieved! Stopping training.")
                control.should_training_stop = True
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss

def train_salary_model_transformers(output_dir):
    """Train salary prediction model using transformers format (like other models)"""
    set_random_seeds(42)
    
    MODEL_ID = "gpt2"
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    if "/kaggle/working" in os.getcwd():
        CACHE_DIR = "/kaggle/working/cache"
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Try to load real data, fallback to synthetic
    real_df, dataset_name = load_real_salary_data()
    
    if real_df is not None:
        logger.info(f"Using real data from {dataset_name}")
        text_data = prepare_salary_text_data(real_df, is_real_data=True, dataset_name=dataset_name)
    else:
        logger.info("Using synthetic salary data")
        synthetic_df = create_synthetic_salary_data(6000)  # Larger dataset
        text_data = prepare_salary_text_data(synthetic_df, is_real_data=False)
    
    # Create datasets
    dataset = Dataset.from_list(text_data)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.config.use_cache = False
        
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise e
    
    # Configure LoRA for salary prediction
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments optimized for salary prediction
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        num_train_epochs=10,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=True,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        dataloader_drop_last=True,
        group_by_length=True,
        save_safetensors=True,
    )
    
    # Callbacks
    callbacks = [
        SalaryLossCallback(target_loss=1.2),
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
        callbacks=callbacks,
    )
    
    logger.info(f"Starting salary model training with {len(train_dataset)} examples")
    
    # Train the model
    trainer.train()
    
    # Save the model (LoRA adapter format - consistent with other models)
    trainer.save_model(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata for consistency
    metadata = {
        "model_type": "salary_prediction",
        "base_model": MODEL_ID,
        "training_data_source": dataset_name if dataset_name else "synthetic",
        "training_samples": len(train_dataset),
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules
        }
    }
    
    with open(os.path.join(output_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Salary model training complete! Model saved to {output_dir}")
    logger.info("Model format: LoRA adapter (consistent with cover_letter and interview models)")

def main():
    parser = argparse.ArgumentParser(description="Train salary prediction model using transformers")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the model")
    args = parser.parse_args()
    
    train_salary_model_transformers(args.output_dir)

if __name__ == "__main__":
    main()