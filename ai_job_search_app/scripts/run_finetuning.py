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
    """Balanced training approach for sub-0.8 loss."""
    set_random_seeds(42)
    
    MODEL_ID = "gpt2"
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    # Set cache for Kaggle
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
    tokenizer.padding_side = "left"
    
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
        lora_config = get_lora_config(r=64, lora_alpha=128, lora_dropout=0.05)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            num_train_epochs=5,
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
            warmup_steps=200,
            lr_scheduler_type="cosine_with_restarts",
            optim="adamw_torch",
            weight_decay=0.01,
            max_grad_norm=0.5,
            seed=42,
            gradient_checkpointing=True,
            label_smoothing_factor=0.1
        )
        
        callbacks = [
            ImprovedTargetLossCallback(target_loss=0.5),
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    else:
        lora_config = get_lora_config()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            num_train_epochs=3,
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

def prepare_interview_data(dataset_name):
    set_random_seeds(42)
    
    behavioral_questions = [
        ("Tell me about yourself", "I am a dedicated professional with [X years] of experience in [field]. My expertise includes [key skills], and I'm passionate about [relevant interest]. I've successfully [achievement], which demonstrates my ability to [relevant skill]."),
        ("Why do you want this position?", "This role aligns perfectly with my career goals and expertise in [relevant area]. I'm excited about [specific aspect of company/role] and believe my experience in [relevant experience] will allow me to make immediate contributions to your team."),
        ("What are your greatest strengths?", "My greatest strengths include [specific skill 1], which I demonstrated when [example]. Additionally, my [skill 2] has enabled me to [achievement]. I also pride myself on my [soft skill], which helps me [benefit to team]."),
        ("What is your biggest weakness?", "I've been working on improving my [skill], as I realized it was limiting my effectiveness in [area]. I've taken steps to address this by [specific action], and I've seen improvement in [specific result]."),
        ("Where do you see yourself in 5 years?", "In five years, I see myself having grown significantly in [skill area], potentially leading [type of projects/team]. I'm interested in developing expertise in [area] and contributing to [company goal] while continuing to learn and take on new challenges."),
        ("Why are you leaving your current job?", "I'm seeking new challenges and opportunities to grow in [specific area]. While I've learned a great deal in my current role, I'm ready for [next step] and believe this position offers the [specific opportunity] I'm looking for."),
        ("Describe a challenging situation you overcame", "I faced a significant challenge when [situation]. I approached it by [action 1], then [action 2]. The result was [positive outcome], and I learned [lesson] which has helped me in subsequent projects."),
        ("How do you handle stress and pressure?", "I manage stress by [strategy 1], which helps me maintain perspective. I also [strategy 2] to stay organized and focused. For example, during [stressful situation], I [specific action] which resulted in [positive outcome]."),
        ("What motivates you?", "I'm motivated by [specific motivator 1], as it allows me to [benefit]. I also find [motivator 2] incredibly rewarding because [reason]. This drives me to consistently [positive behavior] in my work."),
        ("How do you handle conflict with colleagues?", "I approach conflicts by first seeking to understand the other person's perspective through [method]. Then I [action] to find common ground. For instance, when [example situation], I [specific action] which led to [resolution]."),
        ("What makes you unique?", "My unique combination of [skill 1] and [skill 2] allows me to [unique value]. Additionally, my experience in [unique area] gives me a perspective that helps me [benefit]. This has enabled me to [specific achievement]."),
        ("Describe your ideal work environment", "I thrive in environments that [characteristic 1] and [characteristic 2]. I appreciate when [specific aspect] because it allows me to [benefit]. I also value [cultural element] as it aligns with my working style.")
    ]
    
    technical_questions_template = [
        ("Explain {} to a non-technical person", "I would explain {} by comparing it to [everyday analogy]. Essentially, it [simple explanation]. The key benefit is [main advantage], which helps [practical application]."),
        ("How would you optimize a {} system?", "To optimize a {} system, I would first analyze [metric 1] and [metric 2]. Then implement [optimization 1] and [optimization 2]. I'd measure success by tracking [KPI] and iterating based on results."),
        ("What's your experience with {}?", "I have [duration] of hands-on experience with {}. I've used it to [specific application 1] and [application 2]. My most significant project involved [example], where I achieved [result]."),
        ("How do you stay current with {} technologies?", "I stay current by [method 1] and [method 2]. I regularly [activity] and participate in [community/resource]. Recently, I learned about [recent development] which I've started implementing in [context]."),
        ("Describe a {} project you've worked on", "I worked on a {} project that involved [objective]. My role was to [responsibility 1] and [responsibility 2]. The project resulted in [outcome] and taught me [lesson learned]."),
        ("What are best practices for {}?", "Key best practices for {} include [practice 1] to ensure [benefit 1], and [practice 2] for [benefit 2]. I also recommend [practice 3] because it [reason]. These practices have helped me [achievement].")
    ]
    
    skills = ["Python", "JavaScript", "React", "Node.js", "Java", "SQL", "AWS", "Docker", 
              "Kubernetes", "Machine Learning", "API Development", "Database Design",
              "Microservices", "CI/CD", "Agile Methodology", "System Architecture"]
    
    synthetic_data = []
    
    for q, a_template in behavioral_questions:
        for i in range(3):
            answer = a_template.replace("[X years]", f"{random.randint(2,10)} years")
            answer = answer.replace("[field]", random.choice(["software development", "data science", "engineering", "technology"]))
            answer = answer.replace("[key skills]", f"{random.choice(skills)} and {random.choice(skills)}")
            answer = answer.replace("[relevant interest]", random.choice(["solving complex problems", "building scalable systems", "creating innovative solutions"]))
            answer = answer.replace("[achievement]", random.choice(["led successful projects", "improved system performance by 40%", "mentored junior developers"]))
            answer = answer.replace("[relevant skill]", random.choice(["deliver results", "work effectively in teams", "solve complex challenges"]))
            
            placeholders = ["[specific skill 1]", "[skill 2]", "[soft skill]", "[example]", "[area]", "[specific action]", 
                          "[skill area]", "[type of projects/team]", "[company goal]", "[next step]", "[specific opportunity]",
                          "[situation]", "[action 1]", "[action 2]", "[positive outcome]", "[lesson]", "[strategy 1]", 
                          "[strategy 2]", "[stressful situation]", "[specific motivator 1]", "[motivator 2]", "[benefit]",
                          "[method]", "[action]", "[example situation]", "[resolution]", "[unique value]", "[unique area]",
                          "[specific achievement]", "[characteristic 1]", "[characteristic 2]", "[specific aspect]", "[cultural element]"]
            
            for placeholder in placeholders:
                if placeholder in answer:
                    if "skill" in placeholder:
                        answer = answer.replace(placeholder, random.choice(skills), 1)
                    elif "action" in placeholder or "strategy" in placeholder or "method" in placeholder:
                        answer = answer.replace(placeholder, random.choice(["prioritizing tasks", "breaking down problems", "collaborating with teams", "researching solutions"]), 1)
                    elif "outcome" in placeholder or "result" in placeholder or "achievement" in placeholder:
                        answer = answer.replace(placeholder, random.choice(["successful project delivery", "improved efficiency", "positive team dynamics", "exceeded targets"]), 1)
                    else:
                        answer = answer.replace(placeholder, random.choice(["challenging project", "team collaboration", "process improvement", "innovative solutions"]), 1)
            
            synthetic_data.append({
                "text": f"""### Question:
{q}

### Answer:
{answer}
"""
            })
    
    for skill in skills:
        for q_template, a_template in technical_questions_template:
            question = q_template.format(skill)
            answer = a_template.format(skill, skill)
            
            answer = answer.replace("[everyday analogy]", random.choice(["a filing system", "a recipe", "a road network", "a library"]))
            answer = answer.replace("[simple explanation]", f"helps manage and organize {random.choice(['data', 'processes', 'systems', 'workflows'])}")
            answer = answer.replace("[main advantage]", random.choice(["increased efficiency", "better organization", "scalability", "reliability"]))
            answer = answer.replace("[practical application]", random.choice(["solve real problems", "improve performance", "save time", "reduce errors"]))
            answer = answer.replace("[metric 1]", random.choice(["performance", "latency", "throughput", "resource usage"]))
            answer = answer.replace("[metric 2]", random.choice(["scalability", "reliability", "cost", "maintainability"]))
            answer = answer.replace("[optimization 1]", random.choice(["caching", "indexing", "load balancing", "code optimization"]))
            answer = answer.replace("[optimization 2]", random.choice(["parallel processing", "database tuning", "algorithm improvements", "resource allocation"]))
            answer = answer.replace("[KPI]", random.choice(["response time", "error rate", "throughput", "user satisfaction"]))
            answer = answer.replace("[duration]", f"{random.randint(1,5)} years")
            answer = answer.replace("[specific application 1]", random.choice(["build applications", "analyze data", "optimize systems", "solve problems"]))
            answer = answer.replace("[application 2]", random.choice(["improve performance", "automate processes", "enhance features", "debug issues"]))
            answer = answer.replace("[example]", f"building a {random.choice(['web application', 'data pipeline', 'microservice', 'API'])}")
            answer = answer.replace("[result]", random.choice(["50% performance improvement", "reduced costs by 30%", "improved user experience", "automated manual processes"]))
            answer = answer.replace("[method 1]", random.choice(["reading documentation", "following tech blogs", "taking online courses", "attending conferences"]))
            answer = answer.replace("[method 2]", random.choice(["building projects", "contributing to open source", "participating in forums", "experimenting with new features"]))
            answer = answer.replace("[activity]", random.choice(["practice coding", "read articles", "watch tutorials", "review code"]))
            answer = answer.replace("[community/resource]", random.choice(["developer communities", "Stack Overflow", "GitHub projects", "tech meetups"]))
            answer = answer.replace("[recent development]", random.choice(["new framework features", "performance improvements", "security updates", "best practices"]))
            answer = answer.replace("[context]", random.choice(["personal projects", "work assignments", "side projects", "learning exercises"]))
            answer = answer.replace("[objective]", random.choice(["improve performance", "add new features", "fix critical bugs", "migrate systems"]))
            answer = answer.replace("[responsibility 1]", random.choice(["design architecture", "implement features", "write tests", "review code"]))
            answer = answer.replace("[responsibility 2]", random.choice(["optimize performance", "document code", "mentor teammates", "deploy solutions"]))
            answer = answer.replace("[outcome]", random.choice(["successful deployment", "improved metrics", "positive feedback", "system stability"]))
            answer = answer.replace("[lesson learned]", random.choice(["importance of testing", "value of documentation", "benefits of collaboration", "need for planning"]))
            answer = answer.replace("[practice 1]", random.choice(["write clean code", "use version control", "implement testing", "follow standards"]))
            answer = answer.replace("[benefit 1]", random.choice(["maintainability", "reliability", "scalability", "readability"]))
            answer = answer.replace("[practice 2]", random.choice(["regular refactoring", "code reviews", "documentation", "monitoring"]))
            answer = answer.replace("[benefit 2]", random.choice(["code quality", "team knowledge", "system stability", "performance"]))
            answer = answer.replace("[practice 3]", random.choice(["continuous learning", "automation", "security focus", "user feedback"]))
            answer = answer.replace("[reason]", random.choice(["prevents issues", "saves time", "improves quality", "reduces risks"]))
            answer = answer.replace("[achievement]", random.choice(["deliver quality code", "meet deadlines", "exceed expectations", "solve problems efficiently"]))
            
            synthetic_data.append({
                "text": f"""### Question:
{question}

### Answer:
{answer}
"""
            })
    
    print(f"Generated {len(synthetic_data)} interview examples")
    
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

    train_dataset, eval_dataset = prepare_interview_data("synthetic")
    
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
    tokenizer.padding_side = "left"
    
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
        lora_config = get_lora_config(r=64, lora_alpha=128, lora_dropout=0.05)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            num_train_epochs=5,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=30,
            save_strategy="steps",
            save_steps=60,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            warmup_steps=100,
            lr_scheduler_type="polynomial",
            optim="adamw_torch",
            weight_decay=0.05,
            max_grad_norm=0.5,
            seed=42,
            gradient_checkpointing=True,
            label_smoothing_factor=0.1
        )
        
        callbacks = [
            ImprovedTargetLossCallback(target_loss=0.8),
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    else:
        lora_config = get_lora_config()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            num_train_epochs=4,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="epoch",
            report_to="none",
            fp16=True,
            warmup_ratio=0.1,
            seed=42
        )
        callbacks = []
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=384,
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