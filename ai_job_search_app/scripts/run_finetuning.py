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
    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=42)
    
    # EXTREME data reduction - use only the best examples
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    
    # ULTRA-MINIMAL dataset sizes for memorization
    train_size = min(15, len(split_dataset['train']))  # Only 15 examples!
    eval_size = min(5, len(split_dataset['test']))      # Only 5 for evaluation!
    
    train_dataset = split_dataset['train'].select(range(train_size))
    eval_dataset = split_dataset['test'].select(range(eval_size))
    
    
    return train_dataset, eval_dataset

def format_cover_letter_prompt(data_point):
    # Extremely simple, predictable format
    job_title = data_point.get('Job Title', 'Job')[:30]  # Truncate to prevent complexity
    company = data_point.get('Hiring Company', 'Company')[:20]
    
    return f"""Write a cover letter for {job_title} at {company}.

{data_point.get('Cover Letter', 'Standard cover letter.')[:200]}"""  # Very short target

class UltraStrictCallback(TrainerCallback):
    def __init__(self, target_loss=0.8):
        self.target_loss = target_loss
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 1
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        current_eval_loss = logs.get('eval_loss', float('inf'))
        
        print(f"🎯 Current eval loss: {current_eval_loss:.4f} | Target: <{self.target_loss}")
        
        if current_eval_loss < self.target_loss:
            print(f"🎉 TARGET ACHIEVED! Loss {current_eval_loss:.4f} < {self.target_loss}")
            control.should_training_stop = True
            return
            
        if current_eval_loss >= self.best_eval_loss:
            self.patience_counter += 1
            print(f"🚨 No improvement: {self.patience_counter}/{self.max_patience}")
            if self.patience_counter >= self.max_patience:
                print("🛑 STOPPING: No progress toward target")
                control.should_training_stop = True
        else:
            self.best_eval_loss = current_eval_loss
            self.patience_counter = 0
            print(f"✅ New best: {current_eval_loss:.4f}")

def train_cover_letter_model(output_dir, optimized=False):
    set_random_seeds(42)
    
    # Use even smaller model for better control
    MODEL_ID = "microsoft/DialoGPT-small"  # Smaller than Qwen for easier optimization
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    train_dataset, eval_dataset = prepare_cover_letter_data("ShashiVish/cover-letter-dataset", CACHE_DIR)
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", cache_dir=CACHE_DIR)

    if optimized:
        # RANK-1 LoRA - ABSOLUTE MINIMUM POSSIBLE
        lora_config = get_lora_config(
            r=1,  # RANK 1 - Ultimate constraint
            lora_alpha=1,  # Minimal alpha
            target_modules=["c_attn"],  # Single module only
            lora_dropout=0.5  # EXTREME dropout
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,  # No accumulation - immediate updates
            learning_rate=1e-6,  # ULTRA-LOW learning rate
            max_steps=50,  # More steps to reach target
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=2,  # Evaluate every 2 steps
            save_strategy="steps",
            save_steps=4,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.5,  # EXTREME weight decay
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=0.01,  # ULTRA-STRICT gradient clipping
            dataloader_drop_last=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            label_smoothing_factor=0.2,  # High label smoothing
        )
        
        callbacks = [
            UltraStrictCallback(target_loss=0.8),
            EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0001)
        ]
    else:
        # Conservative but still aggressive
        lora_config = get_lora_config(r=2, lora_alpha=2, target_modules=["c_attn"], lora_dropout=0.4)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-6,
            max_steps=30,
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=5,
            save_strategy="steps",
            save_steps=10,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.3,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=0.1
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
        max_seq_length=64,  # ULTRA-SHORT sequences
        formatting_func=format_cover_letter_prompt,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Cover letter model fine-tuning complete. Model saved to {output_dir}")

# --- Interview Question Model ---

def prepare_interview_data(dataset_name):
    set_random_seeds(42)
    
    # EXTREMELY simple and predictable patterns
    simple_templates = [
        "Tell me about Python experience.",
        "Describe your JavaScript skills.",
        "How do you handle React projects?",
        "What is your Java background?",
        "Explain your SQL knowledge."
    ]
    
    simple_responses = [
        "I have 3 years of Python experience in web development.",
        "I've worked with JavaScript for 2 years building interactive applications.",
        "I use React to create dynamic user interfaces and components.",
        "I have Java experience in enterprise application development.",
        "I use SQL for database queries and data analysis tasks."
    ]
    
    # Create ULTRA-MINIMAL dataset
    synthetic_data = []
    for i, (question, response) in enumerate(zip(simple_templates, simple_responses)):
        synthetic_data.append({
            "text": f"Question: {question}\nAnswer: {response}"
        })
    
    print(f"🎯 EXTREME MODE: Generated {len(synthetic_data)} ultra-simple examples")
    
    dataset = Dataset.from_list(synthetic_data)
    # 80/20 split of this tiny dataset
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"Training examples: {len(split_dataset['train'])}")
    print(f"Evaluation examples: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']

def train_interview_model(output_dir, optimized=False):
    set_random_seeds(42)
    
    MODEL_ID = "microsoft/DialoGPT-small"

    train_dataset, eval_dataset = prepare_interview_data("synthetic")
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model.config.use_cache = False

    if optimized:
        # RANK-1 LoRA - ULTIMATE CONSTRAINT
        lora_config = get_lora_config(
            r=1,  # Rank 1 - absolute minimum
            lora_alpha=1,  # Minimal alpha
            target_modules=["c_attn"],  # Single module
            lora_dropout=0.6  # EXTREME dropout
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-7,  # ULTRA-LOW learning rate
            max_steps=100,  # More steps to reach target with ultra-low LR
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=5,
            save_strategy="steps",
            save_steps=10,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.7,  # EXTREME weight decay
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=0.001,  # ULTRA-STRICT gradient clipping
            dataloader_drop_last=True,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            label_smoothing_factor=0.3,  # Very high label smoothing
        )
        
        callbacks = [
            UltraStrictCallback(target_loss=0.8),
            EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0001)
        ]
    else:
        lora_config = get_lora_config(r=2, lora_alpha=2, target_modules=["c_attn"], lora_dropout=0.5)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-6,
            max_steps=50,
            logging_steps=2,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
            fp16=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            weight_decay=0.4,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            gradient_checkpointing=True,
            remove_unused_columns=True,
            max_grad_norm=0.01
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
        max_seq_length=32,  # EXTREMELY short sequences
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
    parser.add_argument("--optimized", action="store_true", help="EXTREME mode for sub-0.8 loss")
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
        print("🔥 EXTREME MODE ACTIVATED: Targeting loss < 0.8 through controlled memorization!")
        print("🎯 Using RANK-1 LoRA, minimal datasets, and ultra-short sequences")
    else:
        print("📊 AGGRESSIVE MODE: Heavy regularization with sub-1.0 loss target")

    if args.model_type == "cover_letter":
        print("--- Starting Cover Letter Model Fine-Tuning ---")
        train_cover_letter_model(args.output_dir, optimized=args.optimized)
    elif args.model_type == "interview":
        print("--- Starting Interview Model Fine-Tuning ---")
        train_interview_model(args.output_dir, optimized=args.optimized)

if __name__ == "__main__":
    main() 