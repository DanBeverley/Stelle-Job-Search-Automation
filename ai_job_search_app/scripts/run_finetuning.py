import torch
import os
import argparse
import warnings
import random
import numpy as np
import time
import logging
from subprocess import run, PIPE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, Dataset, DownloadConfig
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer,
        BitsAndBytesConfig, TrainingArguments, TrainerCallback, EarlyStoppingCallback,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from huggingface_hub import login
    import requests
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def download_with_retry(download_func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            timeout = 30 * (attempt + 1)
            if 'cache_dir' in kwargs:
                kwargs['local_files_only'] = False
            if 'timeout' in kwargs:
                kwargs['timeout'] = timeout
            return download_func(*args, **kwargs)
        except Exception as e:
            if "timeout" in str(e).lower():
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 10)
                else:
                    if 'local_files_only' in kwargs:
                        kwargs['local_files_only'] = True
                        try:
                            return download_func(*args, **kwargs)
                        except:
                            pass
            raise e

def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

def get_lora_config(r=16, lora_alpha=32, lora_dropout=0.05):
    return LoraConfig(
        r=r, lora_alpha=lora_alpha, target_modules=["c_attn", "c_proj"],
        lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM
    )

def create_synthetic_cover_letters(n_samples=500):
    set_random_seeds(42)
    
    job_titles = ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer"]
    companies = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix"]
    
    template = """Dear Hiring Manager,

I am writing to express my interest in the {job_title} position at {company}. With {years} years of experience in software development, I am excited about the opportunity to contribute to your team.

My expertise in {tech_stack} and passion for clean code align with {company}'s commitment to excellence. I am particularly drawn to {company}'s innovative approach to solving complex problems.

Thank you for considering my application.

Best regards,
[Your Name]"""
    
    tech_stacks = [
        "Python, Django, PostgreSQL", "JavaScript, React, Node.js", 
        "Java, Spring Boot, MySQL", "Go, Docker, Kubernetes"
    ]
    
    data = []
    for i in range(n_samples):
        data.append({
            "Job Title": random.choice(job_titles),
            "Hiring Company": random.choice(companies),
            "Cover Letter": template.format(
                job_title=random.choice(job_titles),
                company=random.choice(companies),
                years=random.randint(2, 8),
                tech_stack=random.choice(tech_stacks)
            )
        })
    
    return Dataset.from_list(data)

def load_dataset_with_retry(dataset_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            download_config = DownloadConfig(max_retries=2, resume_download=True)
            dataset = load_dataset(dataset_name, split="train", download_config=download_config)
            return dataset
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 10)
            else:
                raise e

def prepare_cover_letter_data():
    try:
        dataset = load_dataset_with_retry("PolyAI/job-posting-classification")
        if len(dataset) < 100:
            raise ValueError("Dataset too small")
        
        processed_data = []
        for item in dataset:
            if 'job_description' in item and 'job_title' in item:
                prompt = f"Write a cover letter for: {item['job_title']}\nDescription: {item['job_description'][:200]}"
                response = f"Dear Hiring Manager,\n\nI am interested in the {item['job_title']} position..."
                processed_data.append({"text": f"### Human: {prompt}\n\n### Assistant: {response}\n\n### End"})
        
        return Dataset.from_list(processed_data[:1000])
    except:
        logger.warning("Using synthetic cover letter data")
        return create_synthetic_cover_letters()

def create_synthetic_interview_data(n_samples=500):
    set_random_seeds(42)
    
    questions = [
        "Tell me about yourself", "What are your strengths?", "Why do you want this job?",
        "Describe a challenging project", "How do you handle stress?", "Where do you see yourself in 5 years?"
    ]
    
    responses = [
        "I'm a software engineer with {years} years of experience...",
        "My key strength is problem-solving and attention to detail...",
        "I'm excited about this role because it aligns with my career goals...",
        "I recently worked on a project that required {tech} skills...",
        "I handle stress by prioritizing tasks and taking breaks...",
        "In 5 years, I see myself in a senior technical role..."
    ]
    
    data = []
    for i in range(n_samples):
        question = random.choice(questions)
        response = random.choice(responses).format(
            years=random.randint(2, 8),
            tech=random.choice(["Python", "JavaScript", "Java", "React"])
        )
        data.append({"text": f"### Human: {question}\n\n### Assistant: {response}\n\n### End"})
    
    return Dataset.from_list(data)

def prepare_interview_data():
    try:
        dataset = load_dataset_with_retry("microsoft/DialoGPT-medium")
        processed_data = []
        for item in dataset[:1000]:
            if 'text' in item:
                processed_data.append({"text": item['text']})
        return Dataset.from_list(processed_data)
    except:
        logger.warning("Using synthetic interview data")
        return create_synthetic_interview_data()

class EarlyStoppingWithTargetLoss(EarlyStoppingCallback):
    def __init__(self, target_loss=1.5, patience=3):
        super().__init__(early_stopping_patience=patience)
        self.target_loss = target_loss
        self.best_loss = float('inf')

    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        current_loss = logs.get("eval_loss")
        if current_loss and current_loss < self.target_loss:
            logger.info(f"Target loss {self.target_loss} achieved! Current loss: {current_loss}")
            control.should_training_stop = True
        
        if current_loss and current_loss < self.best_loss:
            self.best_loss = current_loss
        
        super().on_evaluate(args, state, control, model, logs, **kwargs)

def train_cover_letter_model(output_dir):
    logger.info("Training cover letter model")
    
    dataset = prepare_cover_letter_data()
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.config.use_cache = False
    
    lora_config = get_lora_config(r=32, lora_alpha=64, lora_dropout=0.1)
    model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        formatting_func=lambda x: x["text"]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_interview_model(output_dir):
    logger.info("Training interview model with ultra-conservative settings")
    
    dataset = prepare_interview_data()
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.config.use_cache = False
    
    lora_config = get_lora_config(r=16, lora_alpha=32, lora_dropout=0.25)
    model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        learning_rate=1e-6,
        num_train_epochs=2,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        formatting_func=lambda x: x["text"],
        callbacks=[EarlyStoppingWithTargetLoss(target_loss=1.8, patience=5)]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_salary_model(output_dir):
    logger.info("Training salary model using existing script")
    
    script_path = os.path.join(os.path.dirname(__file__), "train_salary_model.py")
    
    # Verify script exists
    if not os.path.exists(script_path):
        logger.error(f"Salary training script not found at: {script_path}")
        # Try alternative path
        alt_script_path = "train_salary_model.py"
        if os.path.exists(alt_script_path):
            script_path = alt_script_path
            logger.info(f"Found script at alternative path: {script_path}")
        else:
            raise RuntimeError(f"Salary training script not found: {script_path}")
    
    logger.info(f"Running salary training script: {script_path}")
    try:
        result = run([
            "python", script_path, 
            "--output_dir", output_dir,
            "--format", "transformers"
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode != 0:
            logger.error(f"Salary model training failed with exit code: {result.returncode}")
            if result.stderr:
                logger.error(f"Stderr: {result.stderr}")
            if result.stdout:
                logger.error(f"Stdout: {result.stdout}")
            raise RuntimeError("Salary model training failed")
        
        logger.info("Salary model training completed successfully")
        if result.stdout:
            logger.info(f"Training output: {result.stdout}")
            
    except Exception as e:
        logger.error(f"Exception during salary model training: {e}")
        raise RuntimeError(f"Salary model training failed: {e}")

def main():
    if not IMPORTS_SUCCESSFUL:
        logger.error("Required dependencies not available")
        return
    
    parser = argparse.ArgumentParser(description="Fine-tune models for job search application")
    parser.add_argument("--model_type", choices=["cover_letter", "interview", "salary"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for authentication")
    
    args = parser.parse_args()
    
    if args.hf_token:
        login(token=args.hf_token)
    
    set_random_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_type == "cover_letter":
        train_cover_letter_model(args.output_dir)
    elif args.model_type == "interview":
        train_interview_model(args.output_dir)
    elif args.model_type == "salary":
        train_salary_model(args.output_dir)
    
    logger.info(f"Training completed for {args.model_type} model")

if __name__ == "__main__":
    main()