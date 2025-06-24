import torch
import os
import argparse
import warnings
import random
import numpy as np
import time

# Handle potential import issues with graceful fallbacks
try:
    from datasets import load_dataset, Dataset, DownloadConfig
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        BitsAndBytesConfig, 
        TrainingArguments,
        TrainerCallback,
        EarlyStoppingCallback,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from huggingface_hub import login
    import requests
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

def download_with_retry(download_func, *args, max_retries=3, **kwargs):
    """Generic retry wrapper for download functions"""
    for attempt in range(max_retries):
        try:
            # Increase timeout for each retry
            timeout = 30 * (attempt + 1)
            if 'cache_dir' in kwargs:
                kwargs['local_files_only'] = False
            
            # Try to set longer timeout if possible
            if 'timeout' in kwargs:
                kwargs['timeout'] = timeout
                
            return download_func(*args, **kwargs)
        except Exception as e:
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                print(f"Timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Using offline mode if available.")
                    if 'local_files_only' in kwargs:
                        kwargs['local_files_only'] = True
                        try:
                            return download_func(*args, **kwargs)
                        except:
                            raise e
            else:
                raise e

def get_quantization_config():
    """Returns the BitsAndBytes configuration for 4-bit quantization (QLoRA)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

def get_lora_config(r=16, lora_alpha=32, lora_dropout=0.05):
    """Returns the LoRA configuration."""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
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

def create_synthetic_cover_letters(n_samples=500):
    """Create synthetic cover letter data as fallback"""
    print("Creating synthetic cover letter data...")
    
    job_titles = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager",
                  "Sales Representative", "Project Manager", "Business Analyst", "UX Designer"]
    companies = ["Tech Corp", "Innovation Labs", "Global Solutions", "Future Systems",
                 "Digital Dynamics", "Creative Agency", "StartUp Inc", "Enterprise Co"]
    
    templates = [
        "Dear Hiring Manager,\n\nI am writing to express my strong interest in the {title} position at {company}. With my background in the field and proven track record of success, I am confident I would be a valuable addition to your team.\n\nI bring extensive experience in relevant areas and have consistently delivered results in my previous roles. My skills align perfectly with your requirements, and I am excited about the opportunity to contribute to {company}'s continued success.\n\nI look forward to discussing how I can contribute to your team.\n\nSincerely,\nApplicant",
        "Dear {company} Team,\n\nI am excited to apply for the {title} role at your organization. My professional experience and passion for the industry make me an ideal candidate for this position.\n\nThroughout my career, I have developed strong skills that directly relate to this role. I am particularly drawn to {company}'s innovative approach and would be thrilled to contribute to your mission.\n\nThank you for considering my application. I am eager to bring my expertise to your team.\n\nBest regards,\nApplicant"
    ]
    
    data = []
    for _ in range(n_samples):
        title = random.choice(job_titles)
        company = random.choice(companies)
        template = random.choice(templates)
        letter = template.format(title=title, company=company)
        
        data.append({
            "Job Title": title,
            "Hiring Company": company,
            "Cover Letter": letter
        })
    
    return Dataset.from_list(data)

def prepare_cover_letter_data(dataset_name, cache_dir):
    try:
        download_config = DownloadConfig(
            max_retries=3,
            num_proc=1,
            resume_download=True
        )
        
        dataset = download_with_retry(
            load_dataset,
            dataset_name,
            split="train",
            cache_dir=cache_dir,
            download_config=download_config
        )
        
        dataset = dataset.shuffle(seed=42)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        train_size = min(800, len(split_dataset['train']))
        eval_size = min(100, len(split_dataset['test']))
        
        train_dataset = split_dataset['train'].select(range(train_size))
        eval_dataset = split_dataset['test'].select(range(eval_size))
        
    except Exception as e:
        print(f"Failed to load dataset from HuggingFace: {e}")
        print("Using synthetic data instead...")
        
        dataset = create_synthetic_cover_letters(1000)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    
    def process_example(example):
        job_title = example.get('Job Title', 'Unknown Position')
        company = example.get('Hiring Company', 'Unknown Company')
        cover_letter = example.get('Cover Letter', '').strip()
        
        if len(cover_letter) < 50:
            cover_letter = f"I am writing to apply for the {job_title} position at {company}. I believe my skills and experience make me an excellent candidate for this role."
        
        instruction = f"Write a professional cover letter for the {job_title} position at {company}."
        
        text = f"""### Instruction:
{instruction}

### Response:
{cover_letter}

### End"""
        
        return {"text": text}
    
    train_dataset = train_dataset.map(process_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(process_example, remove_columns=eval_dataset.column_names)
    
    train_dataset = train_dataset.filter(lambda x: len(x['text']) > 100)
    eval_dataset = eval_dataset.filter(lambda x: len(x['text']) > 100)
    
    print(f"Training with {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples")
    
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

class ImprovedTargetLossCallback(TrainerCallback):
    """Callback that stops training when target loss is achieved."""
    def __init__(self, target_loss=1.5):
        self.target_loss = target_loss
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 5
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None or 'eval_loss' not in logs:
            return
            
        current_eval_loss = logs.get('eval_loss', float('inf'))
        
        print(f"Current eval loss: {current_eval_loss:.4f} | Target: <{self.target_loss}")
        
        if current_eval_loss < self.target_loss:
            print(f"Target achieved! Loss {current_eval_loss:.4f} < {self.target_loss}")
            control.should_training_stop = True
            return
            
        if current_eval_loss < self.best_eval_loss:
            self.best_eval_loss = current_eval_loss
            self.patience_counter = 0
            print(f"New best: {current_eval_loss:.4f}")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.max_patience:
                print("Early stopping: No improvement")
                control.should_training_stop = True

def train_cover_letter_model(output_dir, optimized=False):
    set_random_seeds(42)
    
    MODEL_ID = "gpt2"
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    if "/kaggle/working" in os.getcwd():
        CACHE_DIR = "/kaggle/working/cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

    train_dataset, eval_dataset = prepare_cover_letter_data("ShashiVish/cover-letter-dataset", CACHE_DIR)
    
    try:
        tokenizer = download_with_retry(
            GPT2Tokenizer.from_pretrained,
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"Failed to download tokenizer: {e}")
        print("Attempting to use offline mode...")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    try:
        model = download_with_retry(
            GPT2LMHeadModel.from_pretrained,
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("Attempting to use offline mode...")
        model = GPT2LMHeadModel.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
    
    model.config.use_cache = False
    
    if optimized:
        # Adjusted to prevent overfitting with higher dropout and regularization
        lora_config = get_lora_config(r=16, lora_alpha=32, lora_dropout=0.15)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            num_train_epochs=2,  # Reduced epochs to prevent overfitting
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            weight_decay=0.05,  # Increased weight decay
            max_grad_norm=1.0,
            seed=42,
            gradient_checkpointing=False,
            label_smoothing_factor=0.0
        )
        
        callbacks = [
            ImprovedTargetLossCallback(target_loss=1.0),  # Higher target to prevent overfitting
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    else:
        lora_config = get_lora_config()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            num_train_epochs=2,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            report_to="none",
            fp16=True,
            warmup_ratio=0.1,
            seed=42
        )
        callbacks = []
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.train()

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

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Cover letter model fine-tuning complete. Model saved to {output_dir}")

# --- Interview Question Model ---

def prepare_interview_data_improved(dataset_name):
    """Generate higher quality interview data with better structure"""
    set_random_seeds(42)
    
    # Simpler, more focused Q&A pairs
    basic_questions = [
        {
            "q": "Tell me about yourself.",
            "a": "I'm a software engineer with 5 years of experience in full-stack development. I specialize in React and Node.js, and I'm passionate about building scalable web applications. In my current role, I've led several successful projects that improved user engagement by 40%."
        },
        {
            "q": "Why do you want to work here?",
            "a": "I'm impressed by your company's commitment to innovation and the impact your products have on users. The role aligns perfectly with my skills in cloud architecture and my goal to work on large-scale systems. I believe I can contribute significantly to your engineering team."
        },
        {
            "q": "What are your strengths?",
            "a": "My key strengths are problem-solving and communication. I excel at breaking down complex technical problems into manageable solutions. I also have a talent for explaining technical concepts to non-technical stakeholders, which has been valuable in cross-functional projects."
        },
        {
            "q": "What are your weaknesses?",
            "a": "I sometimes spend too much time perfecting code details. I've learned to balance perfectionism with deadlines by setting clear priorities and using code reviews to ensure quality without over-engineering."
        },
        {
            "q": "Where do you see yourself in 5 years?",
            "a": "I see myself as a technical lead or architect, mentoring junior developers and driving technical decisions. I want to deepen my expertise in distributed systems and contribute to open-source projects in my field."
        },
        {
            "q": "Describe a challenging project.",
            "a": "I led the migration of a monolithic application to microservices, serving 1M+ users. The challenge was maintaining zero downtime. We implemented a phased approach with careful testing, resulting in improved performance and easier maintenance."
        },
        {
            "q": "How do you handle pressure?",
            "a": "I handle pressure by staying organized and maintaining clear communication. I break large tasks into smaller milestones and prioritize based on impact. Regular check-ins with the team help identify issues early."
        },
        {
            "q": "Why are you leaving your current job?",
            "a": "I'm looking for new challenges and growth opportunities. While I've learned a lot in my current role, I'm ready to work on larger-scale projects and expand my skills in cloud technologies and system design."
        },
        {
            "q": "What motivates you?",
            "a": "I'm motivated by solving complex problems and seeing the real-world impact of my work. Building products that users love and working with talented teams energizes me. Continuous learning is also a key driver."
        },
        {
            "q": "How do you stay updated with technology?",
            "a": "I follow tech blogs, participate in online communities, and work on side projects. I also attend conferences and take online courses. Recently, I've been exploring Rust and WebAssembly for performance-critical applications."
        }
    ]
    
    # Technical questions for common skills
    technical_templates = [
        {
            "skill": "Python",
            "qa_pairs": [
                ("What's your experience with Python?", "I have 4 years of Python experience, primarily in backend development with Django and FastAPI. I've built RESTful APIs, data pipelines, and automation scripts. I'm comfortable with async programming and have experience with data science libraries."),
                ("How do you optimize Python code?", "I use profiling tools to identify bottlenecks, implement caching strategies, and leverage built-in functions. I also consider using NumPy for numerical operations and multiprocessing for CPU-bound tasks."),
            ]
        },
        {
            "skill": "JavaScript",
            "qa_pairs": [
                ("Describe your JavaScript experience.", "I have extensive experience with modern JavaScript, including ES6+ features. I've built SPAs with React, server-side applications with Node.js, and have strong knowledge of asynchronous programming patterns."),
                ("How do you handle JavaScript performance?", "I focus on minimizing DOM manipulation, using debouncing/throttling for events, implementing lazy loading, and optimizing bundle sizes. I also use performance profiling tools in Chrome DevTools."),
            ]
        },
        {
            "skill": "SQL",
            "qa_pairs": [
                ("What's your SQL expertise?", "I'm proficient in writing complex queries, optimizing database performance, and designing schemas. I have experience with PostgreSQL and MySQL, including stored procedures, indexing strategies, and query optimization."),
                ("How do you optimize SQL queries?", "I analyze query execution plans, create appropriate indexes, avoid N+1 queries, and use EXPLAIN to understand performance. I also consider denormalization when appropriate and implement caching strategies."),
            ]
        },
        {
            "skill": "System Design",
            "qa_pairs": [
                ("How do you approach system design?", "I start by understanding requirements and constraints, then design high-level architecture. I consider scalability, reliability, and maintainability. I use design patterns and focus on loose coupling between components."),
                ("Describe a system you've designed.", "I designed a real-time notification system handling millions of events daily. Used message queues for decoupling, Redis for caching, and implemented circuit breakers for fault tolerance. The system scales horizontally."),
            ]
        }
    ]
    
    synthetic_data = []
    
    # Add basic questions multiple times with slight variations
    for _ in range(5):
        for qa in basic_questions:
            synthetic_data.append({
                "text": f"### Human: {qa['q']}\n\n### Assistant: {qa['a']}\n\n### End"
            })
    
    # Add technical questions
    for tech in technical_templates:
        for q, a in tech["qa_pairs"]:
            for _ in range(3):
                synthetic_data.append({
                    "text": f"### Human: {q}\n\n### Assistant: {a}\n\n### End"
                })
    
    # Add some behavioral questions with STAR format
    star_questions = [
        {
            "q": "Tell me about a time you disagreed with a colleague.",
            "a": "Situation: A colleague wanted to use a new framework for a critical project. Task: I needed to evaluate if it was the right choice. Action: I proposed a proof-of-concept to test both options. Result: We found the existing solution was more suitable, saving time and reducing risk."
        },
        {
            "q": "Describe a time you failed.",
            "a": "Situation: I underestimated the complexity of a data migration project. Task: Had to migrate millions of records with minimal downtime. Action: After initial issues, I redesigned the approach with better testing. Result: Learned the importance of thorough planning and testing, which I apply to all projects now."
        },
        {
            "q": "Give an example of leadership.",
            "a": "Situation: Our team was struggling with a tight deadline. Task: Needed to deliver a feature in 2 weeks. Action: I organized daily standups, broke down tasks, and helped blocked team members. Result: We delivered on time with high quality, and the process improvements continued after the project."
        }
    ]
    
    for qa in star_questions:
        for _ in range(4):
            synthetic_data.append({
                "text": f"### Human: {qa['q']}\n\n### Assistant: {qa['a']}\n\n### End"
            })
    
    print(f"Generated {len(synthetic_data)} high-quality interview examples")
    
    random.shuffle(synthetic_data)
    
    dataset = Dataset.from_list(synthetic_data)
    split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
    
    return split_dataset['train'], split_dataset['test']

def train_interview_model(output_dir, optimized=False):
    set_random_seeds(42)
    
    MODEL_ID = "gpt2"
    
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if "/kaggle/working" in os.getcwd():
        CACHE_DIR = "/kaggle/working/cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

    # Use improved data generation
    train_dataset, eval_dataset = prepare_interview_data_improved("synthetic")
    
    try:
        tokenizer = download_with_retry(
            GPT2Tokenizer.from_pretrained,
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"Failed to download tokenizer: {e}")
        print("Attempting to use offline mode...")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    try:
        model = download_with_retry(
            GPT2LMHeadModel.from_pretrained,
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("Attempting to use offline mode...")
        model = GPT2LMHeadModel.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
    
    model.config.use_cache = False

    if optimized:
        # More aggressive training for interview model
        lora_config = get_lora_config(r=64, lora_alpha=128, lora_dropout=0.1)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,  # Higher learning rate
            num_train_epochs=6,  # More epochs
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            warmup_steps=100,
            lr_scheduler_type="polynomial",
            optim="adamw_torch",
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            gradient_checkpointing=False,
            label_smoothing_factor=0.0
        )
        
        callbacks = [
            ImprovedTargetLossCallback(target_loss=2.0),  # Realistic target
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    else:
        lora_config = get_lora_config(r=32, lora_alpha=64, lora_dropout=0.1)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            num_train_epochs=5,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=40,
            save_strategy="epoch",
            report_to="none",
            fp16=True,
            warmup_ratio=0.1,
            seed=42
        )
        callbacks = []
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.train()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=256,  # Shorter sequences for cleaner training
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
    parser.add_argument("--optimized", action="store_true", help="Optimized mode for better performance")
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
        print("Optimized mode: Enhanced training for better performance")
    else:
        print("Standard mode: Quick training")

    if args.model_type == "cover_letter":
        print("--- Starting Cover Letter Model Fine-Tuning ---")
        train_cover_letter_model(args.output_dir, optimized=args.optimized)
    elif args.model_type == "interview":
        print("--- Starting Interview Model Fine-Tuning ---")
        train_interview_model(args.output_dir, optimized=args.optimized)

if __name__ == "__main__":
    main() 