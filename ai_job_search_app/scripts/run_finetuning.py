import torch
import os
import argparse
import warnings
import random
import numpy as np

# Handle potential import issues with graceful fallbacks
try:
    from datasets import load_dataset, Dataset
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig, 
        TrainingArguments,
        TrainerCallback,
        EarlyStoppingCallback
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from huggingface_hub import login
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Some dependencies may be incompatible. Please check your environment.")
    IMPORTS_SUCCESSFUL = False

# Suppress specific deprecation warnings from TRL
warnings.filterwarnings("ignore", message=".*SFTTrainer.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*packing.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_seq_length.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*dataset_text_field.*", category=UserWarning)

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- Utility Functions ---

def get_quantization_config():
    """Returns the BitsAndBytes configuration for 4-bit quantization (QLoRA)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8
    )

def get_lora_config(r, lora_alpha, target_modules, lora_dropout):
    """Returns the LoRA configuration."""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

# Data augmentation functions
def augment_text(text, augment_prob=0.1):
    """Simple text augmentation to increase data diversity."""
    words = text.split()
    augmented_words = []
    
    for word in words:
        if random.random() < augment_prob:
            # Random word replacement with synonyms (simple approach)
            if word.lower() in ['good', 'great', 'excellent']:
                word = random.choice(['outstanding', 'exceptional', 'remarkable'])
            elif word.lower() in ['experience', 'background']:
                word = random.choice(['expertise', 'knowledge', 'skills'])
        augmented_words.append(word)
    
    return ' '.join(augmented_words)

# --- Cover Letter Model ---

def prepare_cover_letter_data(dataset_name, cache_dir):
    """Balanced dataset preparation for sub-0.8 loss target."""
    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=42)
    
    # Reasonable dataset size - enough to learn patterns but small enough to memorize
    split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
    
    train_size = min(200, len(split_dataset['train']))  # 200 examples - reasonable for memorization
    eval_size = min(40, len(split_dataset['test']))     # 40 for evaluation
    
    train_dataset = split_dataset['train'].select(range(train_size))
    eval_dataset = split_dataset['test'].select(range(eval_size))
    
    print(f"Training with {train_size} examples, evaluating on {eval_size} examples")
    
    return train_dataset, eval_dataset

def format_cover_letter_prompt(data_point):
    """Simplified but effective prompt formatting."""
    job_title = data_point.get('Job Title', 'Job Position')
    company = data_point.get('Hiring Company', 'Company')
    cover_letter = data_point.get('Cover Letter', '')
    
    return f"""### Task: Write a professional cover letter

### Job Details:
Position: {job_title}
Company: {company}

### Cover Letter:
{cover_letter}

### End"""

class TargetLossCallback(TrainerCallback):
    """Callback that stops training when target loss is achieved."""
    def __init__(self, target_loss=0.3):
        self.target_loss = target_loss
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 4
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        # Fix: Check if logs exists and has eval_loss
        if logs is None or 'eval_loss' not in logs:
            return
            
        current_eval_loss = logs.get('eval_loss', float('inf'))
        
        print(f"Current eval loss: {current_eval_loss:.4f} | Target: <{self.target_loss}")
        
        if current_eval_loss < self.target_loss:
            print(f"Target achieved! Loss {current_eval_loss:.4f} < {self.target_loss}")
            control.should_training_stop = True
            return
            
        if current_eval_loss >= self.best_eval_loss:
            self.patience_counter += 1
            print(f"No improvement: {self.patience_counter}/{self.max_patience}")
            if self.patience_counter >= self.max_patience:
                print("Stopping: No progress toward target")
                control.should_training_stop = True
        else:
            self.best_eval_loss = current_eval_loss
            self.patience_counter = 0
            print(f"New best: {current_eval_loss:.4f}")

def train_cover_letter_model(output_dir, optimized=False):
    """Balanced training approach for sub-0.8 loss."""
    set_random_seeds(42)
    
    MODEL_ID = "microsoft/DialoGPT-medium"
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    train_dataset, eval_dataset = prepare_cover_letter_data("ShashiVish/cover-letter-dataset", CACHE_DIR)
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", cache_dir=CACHE_DIR)

    if optimized:
        # Balanced configuration for sub-0.8 loss
        lora_config = get_lora_config(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.05
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            max_steps=100,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.001,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=0.3,
            dataloader_drop_last=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
        )
        
        callbacks = [
            TargetLossCallback(target_loss=0.3),
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.005)
        ]
    else:
        # Standard configuration
        lora_config = get_lora_config(r=16, lora_alpha=32, target_modules=["c_attn", "c_proj"], lora_dropout=0.1)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            max_steps=80,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="steps",
            save_steps=40,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=1.0
        )
        callbacks = []
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="Cover Letter",
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=512,
        formatting_func=format_cover_letter_prompt,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Cover letter model fine-tuning complete. Model saved to {output_dir}")

# --- Interview Question Model ---

def prepare_interview_data(dataset_name):
    """Balanced synthetic data for controlled learning."""
    set_random_seeds(42)
    
    job_roles = ["Software Engineer", "Data Scientist", "DevOps Engineer", "Frontend Developer", 
                 "Backend Developer", "Full Stack Developer", "Machine Learning Engineer", 
                 "Cloud Architect", "Product Manager", "QA Engineer"]
    
    technical_skills = ["Python", "JavaScript", "React", "Node.js", "Java", "SQL", "AWS", 
                       "Docker", "Kubernetes", "Git", "REST APIs", "MongoDB", "PostgreSQL", 
                       "Machine Learning", "Data Analysis", "System Design"]
    
    question_templates = [
        "Tell me about your experience with {}.",
        "How would you approach a {} project?",
        "Describe your {} development process.",
        "What challenges have you faced with {}?",
        "How do you stay updated with {} technologies?",
        "Explain a {} concept to someone new.",
        "What's your preferred {} workflow?",
        "How do you troubleshoot {} issues?",
        "Describe a successful {} implementation.",
        "What {} best practices do you follow?"
    ]
    
    response_templates = [
        "I have {} years of hands-on experience with {} where I've successfully delivered multiple projects. My approach focuses on clean code, testing, and scalable solutions.",
        "I start by understanding requirements, then design the architecture considering scalability and maintainability. I use {} methodologies and ensure proper documentation throughout.",
        "My development process involves planning, iterative development, code reviews, and continuous testing. I prioritize {} principles and maintain high code quality standards.",
        "I've encountered challenges with {} complexity, which I solved by breaking down problems systematically and leveraging {} tools for optimization.",
        "I regularly follow {} blogs, participate in developer communities, contribute to open source, and take online courses to stay current with evolving technologies.",
        "I explain {} by starting with fundamental concepts, using practical examples, and gradually building complexity. Visual aids and hands-on demonstrations help reinforce understanding.",
        "I prefer {} workflows that emphasize collaboration, version control, automated testing, and continuous integration to ensure reliable and efficient development cycles.",
        "I approach {} troubleshooting methodically: reproduce the issue, analyze logs, isolate components, test hypotheses, and implement fixes with proper validation.",
        "I successfully implemented {} by following best practices, conducting thorough testing, monitoring performance, and gathering user feedback for continuous improvement.",
        "I follow {} industry standards including code reviews, documentation, security practices, performance optimization, and maintaining clean, readable, and maintainable code."
    ]
    
    synthetic_data = []
    for role in job_roles:
        for skill in technical_skills[:8]:
            for i, template in enumerate(question_templates[:6]):
                question = template.format(skill)
                response = response_templates[i % len(response_templates)].format(
                    random.choice(["2-3", "3-5", "5+"]), skill, 
                    random.choice(["Agile", "DevOps", "SOLID", "DRY", "KISS"])
                )
                
                synthetic_data.append({
                    "text": f"""### Interview Question:
{question}

### Professional Response:
{response}

### End"""
                })
    
    print(f"Generated {len(synthetic_data)} interview examples")
    
    dataset = Dataset.from_list(synthetic_data)
    split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
    
    print(f"Training examples: {len(split_dataset['train'])}")
    print(f"Evaluation examples: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']

def train_interview_model(output_dir, optimized=False):
    """Balanced training for sub-0.8 loss."""
    set_random_seeds(42)
    
    MODEL_ID = "microsoft/DialoGPT-medium"

    train_dataset, eval_dataset = prepare_interview_data("synthetic")
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model.config.use_cache = False

    if optimized:
        # Balanced configuration for sub-0.8 loss
        lora_config = get_lora_config(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.05
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            max_steps=80,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.001,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=0.3,
            dataloader_drop_last=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
        )
        
        callbacks = [
            TargetLossCallback(target_loss=0.3),
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.005)
        ]
    else:
        lora_config = get_lora_config(r=16, lora_alpha=32, target_modules=["c_attn", "c_proj"], lora_dropout=0.1)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            max_steps=60,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=15,
            save_strategy="steps",
            save_steps=30,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=1.0
        )
        callbacks = []
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=384,
        tokenizer=tokenizer,
        packing=False,
        callbacks=callbacks,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Interview model fine-tuning complete. Model saved to {output_dir}")


# --- Main Execution ---

def main():
    if not IMPORTS_SUCCESSFUL:
        print("ERROR: Required dependencies could not be imported.")
        return
    
    parser = argparse.ArgumentParser(description="Fine-tune a model for a specific task.")
    parser.add_argument("--model_type", type=str, required=True, choices=["cover_letter", "interview"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--optimized", action="store_true", help="Optimized mode for sub-0.3 loss target")
    args = parser.parse_args()

    # Hugging Face Login
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        try:
            login(token=hf_token)
            print("Successfully authenticated with Hugging Face!")
        except Exception as e:
            print(f"Failed to authenticate: {e}")

    if args.optimized:
        print("Optimized mode: Targeting sub-0.3 loss for production quality")
    else:
        print("Standard mode: Conservative training approach")

    if args.model_type == "cover_letter":
        print("--- Starting Cover Letter Model Fine-Tuning ---")
        train_cover_letter_model(args.output_dir, optimized=args.optimized)
    elif args.model_type == "interview":
        print("--- Starting Interview Model Fine-Tuning ---")
        train_interview_model(args.output_dir, optimized=args.optimized)

if __name__ == "__main__":
    main() 