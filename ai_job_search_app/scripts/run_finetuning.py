import torch
import os
import argparse
import warnings
import random
import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
    logger.info("All required dependencies imported successfully")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Some dependencies may be incompatible. Please check your environment")
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
    logger.info(f"Random seeds set to {seed} for reproducibility")

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
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    logger.info(f"Waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
                else:
                    logger.warning("Max retries reached. Attempting offline mode if available")
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
    """Create synthetic cover letter data with much more variety and realism"""
    set_random_seeds(42)
    
    job_titles = [
        "Software Engineer", "Senior Software Engineer", "Full Stack Developer", 
        "Backend Developer", "Frontend Developer", "DevOps Engineer", 
        "Data Scientist", "Machine Learning Engineer", "Product Manager",
        "Technical Lead", "Engineering Manager", "QA Engineer",
        "Mobile Developer", "Web Developer", "Cloud Engineer",
        "Cybersecurity Engineer", "Database Administrator", "Systems Analyst",
        "UI/UX Designer", "Business Analyst", "Project Manager",
        "Software Architect", "Site Reliability Engineer", "Platform Engineer"
    ]
    
    companies = [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix",
        "Uber", "Airbnb", "Spotify", "Stripe", "Shopify", "Slack",
        "Zoom", "Dropbox", "Twitter", "LinkedIn", "Adobe", "Salesforce",
        "Oracle", "IBM", "Tesla", "SpaceX", "Palantir", "Databricks",
        "Snowflake", "Cloudflare", "MongoDB", "Atlassian", "GitHub",
        "GitLab", "Docker", "Kubernetes", "Red Hat", "VMware"
    ]
    
    # More varied and realistic cover letter templates
    cover_letter_templates = [
        # Template 1: Experience-focused
        "Dear Hiring Manager,\n\nI am writing to express my strong interest in the {job_title} position at {company}. With {years} years of experience in software development and a proven track record of building scalable applications, I am excited about the opportunity to contribute to your team.\n\nIn my current role, I have successfully led the development of multiple high-impact projects, including implementing microservices architecture that improved system performance by 40% and reduced deployment time by 60%. My expertise in {tech_stack} and passion for clean, maintainable code align perfectly with {company}'s commitment to technical excellence.\n\nI am particularly drawn to {company} because of your innovative approach to solving complex problems and your reputation for fostering a collaborative engineering culture. I would welcome the opportunity to bring my skills in system design, performance optimization, and team leadership to help drive {company}'s continued success.\n\nThank you for considering my application. I look forward to discussing how I can contribute to your team.\n\nBest regards,\n[Your Name]",
        
        # Template 2: Project-focused  
        "Dear {company} Team,\n\nI am excited to apply for the {job_title} role at {company}. Your recent work on {project_area} particularly resonates with my experience building distributed systems and my passion for creating technology that makes a real impact.\n\nDuring my {years} years in software engineering, I've had the opportunity to work on diverse projects ranging from real-time data processing systems handling millions of events per second to user-facing applications serving millions of users. Most recently, I architected and implemented a recommendation engine that increased user engagement by 35% and contributed to a 20% boost in revenue.\n\nWhat excites me most about {company} is your commitment to innovation and the scale of problems you're solving. I'm particularly interested in contributing to your {focus_area} initiatives, where I believe my background in {relevant_tech} and experience with {methodology} would be valuable.\n\nI would love to discuss how my technical skills and problem-solving approach can help {company} continue to push boundaries and deliver exceptional products.\n\nSincerely,\n[Your Name]",
        
        # Template 3: Culture-focused
        "Hello {company} Hiring Team,\n\nI'm writing to apply for the {job_title} position because {company}'s mission to {mission} aligns perfectly with my values and career aspirations. As a software engineer with {years} years of experience, I'm passionate about building technology that solves real-world problems.\n\nThroughout my career, I've thrived in collaborative environments where continuous learning and innovation are encouraged. At my current company, I've mentored junior developers, led technical discussions, and championed best practices that resulted in 50% fewer production incidents and significantly improved code quality.\n\nI'm particularly excited about {company}'s approach to {company_value} and would love to contribute to your team's success. My experience with {technical_skills} and my commitment to writing clean, tested code would enable me to make meaningful contributions from day one.\n\nI believe that great software is built by great teams, and I'm eager to bring my collaborative spirit and technical expertise to {company}.\n\nBest regards,\n[Your Name]",
        
        # Template 4: Skills-focused
        "Dear Hiring Manager,\n\nI am writing to apply for the {job_title} position at {company}. With expertise in {primary_tech}, {secondary_tech}, and {tertiary_tech}, along with {years} years of hands-on experience, I am confident I can make a significant contribution to your engineering team.\n\nMy technical background includes:\n• Designing and implementing scalable backend systems using {backend_tech}\n• Building responsive frontend applications with {frontend_tech}\n• Optimizing database performance and implementing efficient data pipelines\n• Setting up CI/CD pipelines and monitoring systems for production applications\n\nWhat sets me apart is my ability to bridge the gap between technical implementation and business requirements. I've consistently delivered projects on time while maintaining high code quality standards and collaborating effectively with cross-functional teams.\n\n{company}'s reputation for technical innovation and commitment to engineering excellence makes it an ideal place for me to grow my career while contributing to meaningful projects.\n\nI look forward to the opportunity to discuss how my skills and experience can benefit your team.\n\nThank you for your consideration.\n\n[Your Name]",
        
        # Template 5: Growth-focused
        "Dear {company} Team,\n\nI am thrilled to submit my application for the {job_title} role at {company}. As a passionate software engineer with {years} years of experience, I am excited about the opportunity to contribute to your innovative team while continuing to grow my skills in a dynamic environment.\n\nIn my current role, I have gained extensive experience in {tech_area}, successfully delivering projects that have improved system reliability by 45% and reduced response times by 30%. I am particularly proud of my work on {project_type}, where I collaborated with a team of {team_size} engineers to build a solution that now serves over {user_count} users daily.\n\nWhat attracts me most to {company} is your commitment to pushing technological boundaries and your culture of continuous learning. I am eager to take on new challenges, learn from your talented team, and contribute to building products that have a meaningful impact on users' lives.\n\nI am confident that my technical skills, collaborative approach, and enthusiasm for learning make me a strong fit for this role. I would welcome the opportunity to discuss how I can contribute to {company}'s continued success.\n\nWarm regards,\n[Your Name]"
    ]
    
    # Technical stacks and skills
    tech_stacks = [
        "Python, Django, PostgreSQL, Redis", "JavaScript, React, Node.js, MongoDB",
        "Java, Spring Boot, MySQL, Kafka", "Go, Docker, Kubernetes, AWS",
        "TypeScript, Angular, Express.js, Redis", "C#, .NET Core, SQL Server, Azure",
        "Rust, PostgreSQL, Docker, GraphQL", "Scala, Akka, Cassandra, Spark",
        "Python, FastAPI, PostgreSQL, Elasticsearch", "JavaScript, Vue.js, Firebase, GCP"
    ]
    
    focus_areas = [
        "machine learning", "distributed systems", "cloud infrastructure", "data engineering",
        "mobile development", "web performance", "security", "DevOps automation",
        "microservices", "real-time systems", "data visualization", "API design"
    ]
    
    project_areas = [
        "AI and machine learning", "cloud computing", "developer tools", "data analytics",
        "mobile technology", "web infrastructure", "cybersecurity", "IoT systems",
        "blockchain technology", "edge computing", "automation platforms", "search technology"
    ]
    
    company_missions = [
        "democratize access to information", "connect people worldwide", "accelerate sustainable technology",
        "empower developers", "transform digital commerce", "revolutionize transportation",
        "make work more efficient", "improve healthcare outcomes", "advance financial inclusion",
        "enhance online privacy", "simplify complex workflows", "enable remote collaboration"
    ]
    
    company_values = [
        "engineering excellence", "user-first design", "rapid innovation", "open-source contribution",
        "sustainable development", "inclusive culture", "continuous learning", "data-driven decisions",
        "customer obsession", "technical leadership", "collaborative problem-solving", "quality craftsmanship"
    ]
    
    data = []
    for i in range(n_samples):
        job_title = random.choice(job_titles)
        company = random.choice(companies)
        template = random.choice(cover_letter_templates)
        
        # Generate realistic values
        years = random.randint(2, 12)
        tech_stack = random.choice(tech_stacks)
        focus_area = random.choice(focus_areas)
        project_area = random.choice(project_areas)
        mission = random.choice(company_missions)
        company_value = random.choice(company_values)
        
        # Split tech stack for more specific references
        tech_parts = tech_stack.split(", ")
        primary_tech = tech_parts[0] if len(tech_parts) > 0 else "Python"
        secondary_tech = tech_parts[1] if len(tech_parts) > 1 else "React"
        tertiary_tech = tech_parts[2] if len(tech_parts) > 2 else "PostgreSQL"
        backend_tech = primary_tech
        frontend_tech = secondary_tech
        
        # Additional realistic details
        team_size = random.randint(3, 8)
        user_count = random.choice(["10,000", "50,000", "100,000", "500,000", "1 million"])
        tech_area = random.choice(["backend development", "full-stack development", "cloud engineering", "data engineering"])
        project_type = random.choice(["a distributed caching system", "a real-time analytics platform", "a microservices architecture", "an API gateway"])
        methodology = random.choice(["agile development", "test-driven development", "continuous integration", "DevOps practices"])
        relevant_tech = random.choice(tech_parts[:2])  # Use first 2 tech stack items
        technical_skills = tech_stack
        
        # Fill in the template
        cover_letter = template.format(
            job_title=job_title,
            company=company,
            years=years,
            tech_stack=tech_stack,
            focus_area=focus_area,
            project_area=project_area,
            mission=mission,
            company_value=company_value,
            primary_tech=primary_tech,
            secondary_tech=secondary_tech,
            tertiary_tech=tertiary_tech,
            backend_tech=backend_tech,
            frontend_tech=frontend_tech,
            team_size=team_size,
            user_count=user_count,
            tech_area=tech_area,
            project_type=project_type,
            methodology=methodology,
            relevant_tech=relevant_tech,
            technical_skills=technical_skills
        )
        
        data.append({
            "Job Title": job_title,
            "Hiring Company": company,
            "Cover Letter": cover_letter
        })
    
    logger.info(f"Created {len(data)} synthetic cover letter examples")
    return Dataset.from_list(data)

def load_dataset_with_retry(dataset_name, max_retries=3, timeout_seconds=60):
    """Load dataset with proper retry mechanism for HuggingFace datasets"""
    import time
    import os
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to load dataset '{dataset_name}' (attempt {attempt + 1}/{max_retries})")
            
            # Set environment variable for timeout instead
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout_seconds)
            
            # Create download config without timeout parameter (this was the issue!)
            download_config = DownloadConfig(
                max_retries=2,
                num_proc=1,
                resume_download=True,
                # Remove timeout parameter - it doesn't exist!
            )
            
            # Try different approaches in sequence
            for split_name in ["train", None]:  # Try train split first, then default
                try:
                    dataset = load_dataset(
                        dataset_name, 
                        split=split_name,
                        download_config=download_config,
                        trust_remote_code=True,
                        streaming=False  # Force full download
                    )
                    
                    # Convert to regular dataset if it's iterable
                    if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__'):
                        dataset = Dataset.from_generator(lambda: dataset)
                    
                    if len(dataset) > 0:
                        logger.info(f"Successfully loaded dataset '{dataset_name}' with {len(dataset)} examples")
                        return dataset
                    else:
                        logger.warning(f"Dataset loaded but empty, trying next approach")
                        continue
                        
                except Exception as split_error:
                    logger.warning(f"Failed with split '{split_name}': {split_error}")
                    continue
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            # Check if it's a network/timeout issue
            if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'unreachable']):
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 30  # Exponential backoff: 30s, 60s, 120s
                    logger.info(f"Network issue detected. Waiting {wait_time}s before next attempt")
                    time.sleep(wait_time)
                    continue
            
            # For other errors, try offline mode if available
            if attempt == max_retries - 1:
                try:
                    logger.info("Trying offline mode as final attempt")
                    dataset = load_dataset(dataset_name, split="train", local_files_only=True)
                    if len(dataset) > 0:
                        logger.info(f"Loaded from cache: {len(dataset)} examples")
                        return dataset
                except:
                    pass
            
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                logger.info(f"Waiting {wait_time}s before next attempt")
                time.sleep(wait_time)
            else:
                logger.warning(f"All {max_retries} attempts failed. Will use synthetic data instead")
                raise e

def load_real_cover_letter_data_alternative():
    """Try alternative methods to load real cover letter data"""
    alternative_datasets = [
        "ShashiVish/cover-letter-dataset",
        "susanQQ/cover-letter-dataset", 
        "cover-letter-dataset/cover-letter-dataset",
        "career-datasets/cover-letters"
    ]
    
    for dataset_name in alternative_datasets:
        try:
            logger.info(f"Trying alternative dataset: {dataset_name}")
            dataset = load_dataset_with_retry(dataset_name, max_retries=2, timeout_seconds=120)
            
            # Check if dataset has reasonable structure
            if len(dataset) > 10:  # At least 10 examples
                first_example = dataset[0]
                # Look for relevant columns
                relevant_cols = []
                for col in first_example.keys():
                    if any(keyword in col.lower() for keyword in ['cover', 'letter', 'text', 'content']):
                        relevant_cols.append(col)
                
                if relevant_cols:
                    logger.info(f"Found viable dataset: {dataset_name} with {len(dataset)} examples")
                    logger.info(f"Relevant columns: {relevant_cols}")
                    return dataset, dataset_name
                    
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            continue
    
    return None, None

def prepare_cover_letter_data(dataset_name, cache_dir):
    # First, try the main dataset
    dataset = None
    actual_dataset_name = dataset_name
    use_synthetic = False
    
    try:
        # Use improved dataset loading with retry
        dataset = load_dataset_with_retry(dataset_name, max_retries=3, timeout_seconds=120)
        logger.info(f"Successfully loaded primary dataset: {dataset_name}")
        
    except Exception as e:
        logger.error(f"Failed to load primary dataset: {e}")
        
        # Try alternative datasets
        logger.info("Searching for alternative datasets")
        dataset, actual_dataset_name = load_real_cover_letter_data_alternative()
        
        if dataset is None:
            logger.warning("All real datasets failed. Using synthetic data")
            use_synthetic = True
    
    if use_synthetic:
        # Use synthetic data as last resort
        dataset = create_synthetic_cover_letters(1200)  # Larger synthetic dataset
        split_dataset = dataset.train_test_split(test_size=0.12, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        logger.info(f"Using enhanced synthetic dataset: {len(train_dataset)} train, {len(eval_dataset)} eval examples")
    else:
        # Process real dataset
        logger.info(f"Processing real dataset: {actual_dataset_name}")
        
        # Shuffle and split the real dataset
        dataset = dataset.shuffle(seed=42)
        
        # Use larger portion of real data if available
        total_size = len(dataset)
        train_size = min(1500, int(total_size * 0.85))  # Use up to 1500 training examples
        
        if total_size > train_size:
            # Split if we have enough data
            split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
            train_dataset = split_dataset['train'].select(range(min(train_size, len(split_dataset['train']))))
            eval_dataset = split_dataset['test'].select(range(min(200, len(split_dataset['test']))))
        else:
            # Use most for training if dataset is small
            split_point = max(10, int(total_size * 0.1))  # Reserve at least 10 for eval
            train_dataset = dataset.select(range(total_size - split_point))
            eval_dataset = dataset.select(range(total_size - split_point, total_size))
        
        logger.info(f"Using real dataset: {len(train_dataset)} train, {len(eval_dataset)} eval examples")
    
    # Improved data processing function
    def process_example_advanced(example):
        # Try different column name patterns for cover letters
        cover_letter_text = ""
        job_title = "Software Engineer"  # Default
        company = "Technology Company"   # Default
        
        # Look for cover letter content in various column names
        possible_text_cols = ['Cover Letter', 'cover_letter', 'text', 'content', 'letter', 'body']
        for col in possible_text_cols:
            if col in example and example[col] and len(str(example[col]).strip()) > 30:
                cover_letter_text = str(example[col]).strip()
                break
        
        # Look for job title
        possible_job_cols = ['Job Title', 'job_title', 'title', 'position', 'role']
        for col in possible_job_cols:
            if col in example and example[col]:
                job_title = str(example[col]).strip()
                break
        
        # Look for company
        possible_company_cols = ['Hiring Company', 'company', 'organization', 'employer']
        for col in possible_company_cols:
            if col in example and example[col]:
                company = str(example[col]).strip()
                break
        
        # If no cover letter found, create a template
        if not cover_letter_text or len(cover_letter_text) < 50:
            cover_letter_text = f"""Dear Hiring Manager,

I am writing to express my strong interest in the {job_title} position at {company}. With my background in software development and passion for technology, I believe I would be a valuable addition to your team.

In my previous experience, I have developed strong skills in programming, problem-solving, and teamwork. I am particularly drawn to {company} because of its reputation for innovation and commitment to excellence.

I would welcome the opportunity to discuss how my skills and enthusiasm can contribute to your team's success. Thank you for considering my application.

Sincerely,
[Your Name]"""
        
        # Create structured prompt for better training
        instruction = f"Write a professional cover letter for the {job_title} position at {company}."
        
        # Use improved formatting for better model training
        text = f"""### Instruction:
{instruction}

### Response:
{cover_letter_text}

### End"""
        
        return {"text": text}
    
    # Apply processing to both datasets
    train_dataset = train_dataset.map(process_example_advanced, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(process_example_advanced, remove_columns=eval_dataset.column_names)
    
    # Filter out examples that are too short (better quality control)
    train_dataset = train_dataset.filter(lambda x: len(x['text']) > 150)
    eval_dataset = eval_dataset.filter(lambda x: len(x['text']) > 150)
    
    logger.info(f"Final processed dataset sizes: {len(train_dataset)} training examples, {len(eval_dataset)} evaluation examples")
    
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
        
        logger.info(f"Current eval loss: {current_eval_loss:.4f} | Target: <{self.target_loss}")
        
        if current_eval_loss < self.target_loss:
            logger.info(f"Target achieved! Loss {current_eval_loss:.4f} < {self.target_loss}")
            control.should_training_stop = True
            return
            
        if current_eval_loss < self.best_eval_loss:
            self.best_eval_loss = current_eval_loss
            self.patience_counter = 0
            logger.info(f"New best: {current_eval_loss:.4f}")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.max_patience:
                logger.info("Early stopping: No improvement")
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
        lora_config = get_lora_config(r=64, lora_alpha=128, lora_dropout=0.1)  # Higher capacity for better learning
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,  # Slightly larger batch
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,  # Moderate accumulation
            learning_rate=2e-5,  # Optimal learning rate for cover letters
            num_train_epochs=12,  # More epochs for thorough learning
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=50,  # More frequent evaluation
            save_strategy="steps",
            save_steps=100,
            save_total_limit=8,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            warmup_steps=300,  # Proper warmup
            lr_scheduler_type="cosine",  # Better learning rate schedule
            optim="adamw_torch",
            weight_decay=0.01,  # Moderate regularization
            max_grad_norm=1.0,  # Standard gradient clipping
            seed=42,
            gradient_checkpointing=False,
            label_smoothing_factor=0.0,
            dataloader_drop_last=True,
            remove_unused_columns=True,
            prediction_loss_only=True,
            dataloader_num_workers=0,
            save_safetensors=True,
            disable_tqdm=False,
            eval_accumulation_steps=4,
            save_on_each_node=False,
            # Advanced optimization settings
            dataloader_pin_memory=True,
            group_by_length=True,  # Group similar length sequences
            logging_first_step=True,
            log_level="info",
        )
        
        callbacks = [
            ImprovedTargetLossCallback(target_loss=0.8),  # Much more aggressive target
            EarlyStoppingCallback(early_stopping_patience=15)  # Reasonable patience
        ]
    else:
        # Standard optimized configuration
        lora_config = get_lora_config(r=32, lora_alpha=64, lora_dropout=0.15)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,  # Larger standard batch
            gradient_accumulation_steps=8,
            learning_rate=3e-5,  # Good standard rate
            num_train_epochs=8,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            report_to="none",
            fp16=True,
            warmup_ratio=0.15,
            seed=42,
            weight_decay=0.01,
            max_grad_norm=1.0,
            group_by_length=True,
        )
        callbacks = [ImprovedTargetLossCallback(target_loss=1.2)]
    
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

    logger.info(f"Starting {'ultra-optimized' if optimized else 'standard'} cover letter model training")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Cover letter model fine-tuning complete. Model saved to {output_dir}")

def train_ultra_optimized_cover_letter_model(output_dir):
    """Ultra-optimized training specifically designed to achieve very low loss"""
    set_random_seeds(42)
    
    MODEL_ID = "gpt2"
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    if "/kaggle/working" in os.getcwd():
        CACHE_DIR = "/kaggle/working/cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

    # Get the best possible dataset
    train_dataset, eval_dataset = prepare_cover_letter_data("ShashiVish/cover-letter-dataset", CACHE_DIR)
    
    try:
        tokenizer = download_with_retry(
            GPT2Tokenizer.from_pretrained,
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"Failed to download tokenizer: {e}")
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
        model = GPT2LMHeadModel.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
    
    model.config.use_cache = False

    lora_config = get_lora_config(r=128, lora_alpha=256, lora_dropout=0.05)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Small batch for precise gradients
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,  # Large effective batch size
        learning_rate=1.5e-5,  # Sweet spot for this task
        num_train_epochs=20,  # Many epochs for thorough learning
        logging_steps=3,  # Very frequent logging
        eval_strategy="steps",
        eval_steps=25,  # Very frequent evaluation
        save_strategy="steps",
        save_steps=50,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=True,
        warmup_steps=500,  # Long warmup for stability
        lr_scheduler_type="cosine_with_restarts",  # Advanced scheduler
        optim="adamw_torch",
        weight_decay=0.005,  # Light regularization
        max_grad_norm=0.5,  # Tight gradient clipping
        seed=42,
        gradient_checkpointing=False,
        label_smoothing_factor=0.0,
        dataloader_drop_last=True,
        remove_unused_columns=True,
        prediction_loss_only=True,
        dataloader_num_workers=0,
        save_safetensors=True,
        disable_tqdm=False,
        eval_accumulation_steps=2,
        save_on_each_node=False,
        # Ultra-optimization settings
        dataloader_pin_memory=True,
        group_by_length=True,
        logging_first_step=True,
        log_level="info",
        skip_memory_metrics=False,
        # Advanced settings for best performance
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_kwargs={"num_cycles": 2},  # For cosine with restarts
    )
    
    callbacks = [
        ImprovedTargetLossCallback(target_loss=0.5),  # Extremely low target
        EarlyStoppingCallback(early_stopping_patience=25)  # More patience for ultra-optimization
    ]
    
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

    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Evaluation examples: {len(eval_dataset)}")
    
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Cover letter model trained. Model saved to {output_dir}")

# Interview Question Model 

def prepare_interview_data_comprehensive(dataset_name):
    """Generate comprehensive interview data with much more variety"""
    set_random_seeds(42)
    
    # Expanded behavioral questions with more variety
    behavioral_questions = [
        {
            "q": "Tell me about yourself.",
            "variations": [
                "I'm a software engineer with 5 years of experience in full-stack development. I specialize in React and Node.js, and I'm passionate about building scalable web applications. In my current role, I've led several successful projects that improved user engagement by 40%.",
                "I'm a passionate developer with expertise in modern web technologies. I have 4 years of experience building enterprise applications and have led cross-functional teams. I'm particularly interested in performance optimization and user experience.",
                "I'm a full-stack developer with a strong background in JavaScript and Python. I've worked on everything from small startups to large-scale enterprise systems. I enjoy solving complex problems and mentoring junior developers.",
                "I'm an experienced software engineer specializing in backend systems and API design. I have 6 years of experience working with microservices and cloud technologies. I'm passionate about writing clean, maintainable code.",
                "I'm a versatile developer with experience across multiple technologies and domains. I've worked in fintech, healthcare, and e-commerce, always focusing on delivering high-quality solutions that meet business needs.",
                # Additional variations for more diversity
                "I'm a senior software engineer with 8 years of experience building distributed systems and cloud applications. I've led engineering teams of 5-10 people and have deep expertise in both frontend and backend technologies. I'm particularly interested in system architecture and performance optimization.",
                "I'm a data-driven developer with 6 years of experience in both web development and data science. I specialize in building applications that handle large datasets and provide actionable insights. I've worked extensively with Python, JavaScript, and machine learning frameworks.",
                "I'm a full-stack engineer with expertise in modern web technologies and DevOps practices. I have 7 years of experience building scalable applications and implementing CI/CD pipelines. I'm passionate about code quality, testing, and automation."
            ]
        },
        {
            "q": "Why do you want to work here?",
            "variations": [
                "I'm impressed by your company's commitment to innovation and the impact your products have on millions of users. The role aligns perfectly with my skills in cloud architecture and my goal to work on large-scale distributed systems.",
                "Your company's reputation for technical excellence and engineering culture really appeals to me. I'm excited about the opportunity to work on challenging problems at scale and contribute to products that make a real difference.",
                "I've been following your company's work in AI and machine learning, and I'm excited about the opportunity to contribute to these cutting-edge projects. The role matches my background and career aspirations perfectly.",
                "Your company's focus on engineering excellence and innovation is exactly what I'm looking for. I want to work with talented engineers on meaningful projects that have global impact.",
                "I'm drawn to your company's mission and the opportunity to work on products that solve real problems. The technical challenges you face align with my interests in scalable systems and performance optimization.",
                # Additional variations
                "I've always admired your company's approach to building products that truly serve users' needs. The role offers the perfect combination of technical challenges and meaningful impact that I'm looking for in my next position.",
                "Your company's commitment to open source and developer community really resonates with me. I want to work somewhere that values knowledge sharing and continuous learning, and I see that culture here.",
                "I'm particularly excited about your company's work in emerging technologies. The opportunity to work on cutting-edge projects while learning from industry experts is exactly what I need to advance my career."
            ]
        },
        {
            "q": "What are your strengths?",
            "variations": [
                "My key strengths are problem-solving and system design. I excel at breaking down complex technical problems into manageable components and designing scalable solutions. I also have strong communication skills that help me work effectively with cross-functional teams.",
                "I'm particularly strong in debugging and optimization. I have a methodical approach to identifying performance bottlenecks and finding root causes of issues. I also pride myself on writing clean, well-documented code.",
                "My strengths include technical leadership and mentoring. I enjoy helping junior developers grow and leading technical discussions. I'm also skilled at translating business requirements into technical solutions.",
                "I have strong analytical skills and attention to detail. I'm good at seeing the big picture while also focusing on implementation details. I also have experience with agile methodologies and project management.",
                "My main strengths are adaptability and continuous learning. I quickly pick up new technologies and frameworks. I also have strong collaboration skills and work well in diverse, multicultural teams.",
                # Additional variations
                "I'm particularly strong at code architecture and design patterns. I have a good eye for identifying when code can be refactored for better maintainability and I enjoy creating reusable components and libraries.",
                "My strengths include cross-functional collaboration and stakeholder management. I'm good at understanding business requirements and translating them into technical specifications that development teams can execute on.",
                "I excel at performance optimization and scalability challenges. I have experience with profiling applications, identifying bottlenecks, and implementing solutions that can handle increased load and user growth."
            ]
        },
        {
            "q": "Describe a challenging project you worked on.",
            "variations": [
                "I led the migration of a legacy monolithic system to microservices architecture for a high-traffic e-commerce platform. The main challenge was ensuring zero downtime during the transition while maintaining data consistency. We used blue-green deployment and feature flags, resulting in 50% improved performance.",
                "I worked on optimizing a real-time analytics system that was struggling with latency issues. The challenge was processing millions of events per second while maintaining sub-100ms response times. I redesigned the data pipeline using Kafka and Redis, achieving 80% latency reduction.",
                "I built a distributed task processing system that needed to handle varying workloads efficiently. The challenge was auto-scaling based on queue depth while maintaining cost efficiency. I implemented a custom scheduler that reduced processing time by 60% and cut costs by 40%.",
                "I led the development of a new authentication system that needed to integrate with multiple legacy systems. The challenge was maintaining security while ensuring smooth user experience. We implemented OAuth 2.0 with custom adapters, improving security while reducing login friction.",
                "I worked on a data migration project involving petabytes of data with strict downtime requirements. The challenge was ensuring data integrity while minimizing business impact. We developed a custom ETL pipeline that completed the migration 2 weeks ahead of schedule.",
                # Additional variations
                "I architected and implemented a real-time notification system that needed to handle millions of push notifications daily. The challenge was ensuring reliable delivery across multiple platforms while maintaining low latency. I designed a fault-tolerant system using message queues and circuit breakers.",
                "I led the implementation of a new search engine for our e-commerce platform. The challenge was improving search relevance while maintaining fast response times. I implemented Elasticsearch with custom ranking algorithms, resulting in 35% improved conversion rates.",
                "I worked on a critical performance optimization project for our main application. The system was experiencing frequent timeouts under heavy load. I implemented caching strategies, optimized database queries, and redesigned the architecture, resulting in 70% improvement in response times."
            ]
        }
    ]
    
    # Expanded technical deep-dive questions
    technical_deep_dive = [
        {
            "topic": "System Design",
            "questions": [
                ("How would you design a URL shortener like bit.ly?", "I'd start with requirements analysis: handling millions of URLs, fast redirects, and analytics. The core components would be a URL encoding service, database for mappings, cache layer for performance, and analytics service. I'd use base62 encoding for short URLs, Redis for caching hot URLs, and a load balancer for high availability."),
                ("Design a chat application for millions of users.", "Key components: WebSocket servers for real-time messaging, message queue for reliability, user service for authentication, and database for message persistence. I'd use horizontal sharding for the database, implement message acknowledgments, and use CDN for media sharing. Load balancing would distribute WebSocket connections."),
                ("How would you design a recommendation system?", "I'd use a hybrid approach combining collaborative filtering and content-based filtering. The system would have data collection, feature engineering, model training, and serving components. I'd implement A/B testing for model evaluation and use real-time user interactions to update recommendations continuously."),
                ("Design a social media feed system.", "I'd design a system with user service, post service, feed generation service, and notification service. For feed generation, I'd use a pull model for active users and push model for less active ones. I'd implement caching at multiple levels and use CDN for media content. Database would be sharded by user ID."),
                ("How would you design a ride-sharing service like Uber?", "Core components: user service, driver service, trip service, location service, and payment service. I'd use geospatial indexing for location queries, implement real-time tracking with WebSockets, and use event-driven architecture for trip state management. Machine learning would optimize driver-rider matching."),
                ("Design a video streaming platform like YouTube.", "Key components: video upload service, encoding service, metadata service, search service, and content delivery network. I'd use distributed storage for videos, implement adaptive bitrate streaming, and use machine learning for recommendations. Global CDN would ensure low latency worldwide."),
            ]
        },
        {
            "topic": "Algorithms & Data Structures",
            "questions": [
                ("Explain how you would implement a LRU cache.", "I'd use a combination of HashMap and doubly-linked list. The HashMap provides O(1) access, while the linked list maintains the order of access. When accessing an item, I move it to the head. When the cache is full, I remove from the tail. Both get and put operations are O(1)."),
                ("How would you find the median in a stream of integers?", "I'd use two heaps: a max heap for the smaller half and a min heap for the larger half. I maintain the property that the max heap size is either equal to or one more than the min heap size. The median is either the top of the max heap or the average of both heap tops."),
                ("Describe an efficient way to detect cycles in a linked list.", "I'd use Floyd's cycle detection algorithm (tortoise and hare). Use two pointers: slow moves one step, fast moves two steps. If there's a cycle, they'll eventually meet. If fast reaches null, there's no cycle. This is O(n) time and O(1) space."),
                ("How would you implement a thread-safe counter?", "I'd use atomic operations or synchronization mechanisms like locks. For high-performance scenarios, I might use compare-and-swap operations or lock-free data structures. The choice depends on contention levels and performance requirements."),
                ("Explain how to implement a distributed hash table.", "I'd use consistent hashing to distribute keys across nodes. Each node would be responsible for range of hash values. For fault tolerance, I'd replicate data across multiple nodes. When nodes join or leave, I'd redistribute only the affected range of keys."),
                ("How would you design a rate limiter?", "I'd use algorithms like token bucket or sliding window. Token bucket allows burst traffic while maintaining average rate. I'd implement it using Redis with atomic operations for distributed systems. Different rate limits could be applied per user, IP, or API key."),
            ]
        },
        {
            "topic": "Database Design",
            "questions": [
                ("How would you optimize a slow database query?", "I'd start by analyzing the execution plan to identify bottlenecks. Common optimizations include adding indexes, rewriting queries to avoid full table scans, partitioning large tables, and denormalizing for read-heavy workloads. I'd also consider caching frequently accessed data."),
                ("Explain database indexing strategies.", "Indexes speed up read operations but slow down writes. I'd use B-tree indexes for range queries, hash indexes for equality checks, and composite indexes for multi-column searches. I'd avoid over-indexing and regularly analyze index usage to remove unused ones."),
                ("How would you design a database schema for an e-commerce system?", "I'd have tables for users, products, categories, orders, order_items, and payments. I'd normalize to reduce redundancy but denormalize for performance where needed. I'd use foreign keys for referential integrity and indexes on commonly queried columns."),
                ("Explain ACID properties in databases.", "Atomicity ensures transactions are all-or-nothing. Consistency maintains data integrity. Isolation prevents transactions from interfering with each other. Durability ensures committed transactions survive system failures. These properties are crucial for data reliability."),
            ]
        }
    ]
    
    # Expanded behavioral STAR format questions
    star_questions = [
        {
            "q": "Tell me about a time you had to learn a new technology quickly.",
            "a": "Situation: Our team needed to migrate to GraphQL but no one had experience with it. Task: I had to become proficient in 2 weeks to lead the migration. Action: I created a learning plan, built several prototypes, and shared knowledge with the team through daily sessions. Result: We successfully migrated all APIs to GraphQL, improving query efficiency by 30% and reducing over-fetching."
        },
        {
            "q": "Describe a situation where you had to work with a difficult team member.",
            "a": "Situation: A senior developer was resistant to code review feedback and defensive about their code. Task: I needed to maintain team harmony while ensuring code quality. Action: I scheduled a private conversation to understand their concerns and worked together to establish clear code standards. Result: They became more receptive to feedback and our code review process improved significantly."
        },
        {
            "q": "Tell me about a time you made a mistake and how you handled it.",
            "a": "Situation: I accidentally deployed a breaking change to production that affected 20% of users. Task: I needed to fix the issue quickly and prevent future occurrences. Action: I immediately rolled back the deployment, communicated with stakeholders, and implemented better testing procedures. Result: We reduced deployment issues by 90% and I became more careful about testing edge cases."
        },
        {
            "q": "Describe a time you improved a process or system.",
            "a": "Situation: Our deployment process was taking 2 hours and often failed. Task: I was asked to improve deployment reliability and speed. Action: I automated the entire pipeline using CI/CD, added comprehensive testing, and implemented rollback mechanisms. Result: Deployment time reduced to 15 minutes with 99% success rate, and developer productivity increased significantly."
        },
        {
            "q": "Tell me about a time you had to meet a tight deadline.",
            "a": "Situation: We had to deliver a critical feature for a major client in half the planned time due to a changed launch date. Task: I needed to lead the team to deliver quality code under pressure. Action: I broke down the work into smaller tasks, coordinated parallel development, and set up daily standups for quick issue resolution. Result: We delivered the feature on time with full functionality and minimal bugs."
        },
        {
            "q": "Describe a situation where you had to convince others of your technical approach.",
            "a": "Situation: The team wanted to use a popular but resource-heavy framework for our new project. Task: I needed to convince them that a lighter alternative would be better for our use case. Action: I created a proof of concept, analyzed performance metrics, and presented a detailed comparison. Result: The team adopted my approach, resulting in 40% faster load times and reduced infrastructure costs."
        },
        {
            "q": "Tell me about a time you had to deal with ambiguous requirements.",
            "a": "Situation: I was asked to build a 'flexible reporting system' with minimal specifications. Task: I needed to clarify requirements and deliver a useful solution. Action: I interviewed stakeholders, created mockups, and iteratively refined the requirements through feedback sessions. Result: We delivered a configurable reporting system that met all stakeholder needs and is still in use today."
        },
        {
            "q": "Describe a time you mentored someone.",
            "a": "Situation: A junior developer joined our team with limited experience in our tech stack. Task: I was asked to help them become productive quickly. Action: I created a learning plan, paired with them on tasks, and provided regular feedback and encouragement. Result: They became a valuable team member within 3 months and now mentors other junior developers."
        }
    ]
    
    # Expanded industry-specific questions
    industry_questions = [
        ("How do you ensure code quality?", "I use multiple approaches: comprehensive testing including unit, integration, and end-to-end tests; code reviews with clear standards; static analysis tools; and continuous integration. I also practice TDD when appropriate and maintain high test coverage. Documentation and clean code principles are equally important."),
        ("How do you handle technical debt?", "I track technical debt systematically and prioritize it based on business impact. I allocate 20% of sprint capacity to address debt, advocate for refactoring during feature development, and maintain a debt register. I also ensure stakeholders understand the long-term costs of accumulating debt."),
        ("Describe your debugging process.", "I start by reproducing the issue consistently, then gather relevant logs and stack traces. I use binary search to narrow down the problem area and leverage debugging tools and profilers. I document findings and implement fixes with proper testing. Prevention is key, so I also analyze root causes."),
        ("How do you stay current with technology trends?", "I follow tech blogs, participate in developer communities, attend conferences, and work on side projects. I also contribute to open source when possible and take online courses. I focus on understanding fundamentals rather than chasing every new framework."),
        ("How do you approach performance optimization?", "I start by establishing baselines and identifying bottlenecks through profiling. I prioritize optimizations based on impact and implement them incrementally. I consider all layers: database queries, application logic, caching, and infrastructure. Monitoring is crucial to measure improvements."),
        ("Explain your approach to API design.", "I follow RESTful principles and design APIs that are intuitive and consistent. I use proper HTTP methods and status codes, implement versioning from the start, and provide comprehensive documentation. I also consider pagination, rate limiting, and authentication from the beginning."),
        ("How do you handle production incidents?", "I follow a structured incident response process: immediate mitigation to reduce impact, root cause analysis to understand what happened, and implementation of preventive measures. Communication with stakeholders throughout is crucial. I also conduct post-mortems to learn from incidents."),
        ("What's your approach to testing?", "I use a testing pyramid approach: many unit tests, fewer integration tests, and minimal end-to-end tests. I write tests first when doing TDD, and I ensure tests are fast, reliable, and maintainable. I also use mocking appropriately and test edge cases."),
        ("How do you approach software architecture decisions?", "I consider multiple factors: scalability requirements, team expertise, maintenance burden, and business constraints. I document architectural decisions and their rationale. I prefer simple solutions that can evolve over time rather than over-engineered systems."),
        ("Describe your experience with cloud technologies.", "I have experience with AWS/Azure/GCP services including compute, storage, databases, and networking. I understand infrastructure as code, containerization with Docker/Kubernetes, and serverless architectures. I also have experience with monitoring and logging in cloud environments.")
    ]
    
    synthetic_data = []
    
    # Generate behavioral questions with multiple variations (increased multiplier)
    for qa in behavioral_questions:
        for variation in qa["variations"]:
            for _ in range(15):  # Much more repetitions for behavioral questions
                synthetic_data.append({
                    "text": f"### Human: {qa['q']}\n\n### Assistant: {variation}\n\n### End"
                })
    
    # Generate technical deep-dive questions (increased multiplier)
    for topic_data in technical_deep_dive:
        for q, a in topic_data["questions"]:
            for i in range(25):  # Much more repetitions for technical questions
                synthetic_data.append({
                    "text": f"### Human: {q}\n\n### Assistant: {a}\n\n### End"
                })
    
    # Generate STAR format questions (increased multiplier)
    for qa in star_questions:
        for _ in range(30):  # Much more STAR examples
            synthetic_data.append({
                "text": f"### Human: {qa['q']}\n\n### Assistant: {qa['a']}\n\n### End"
            })
    
    # Generate industry questions (increased multiplier)
    for q, a in industry_questions:
        for _ in range(40):  # Much more industry-specific questions
            synthetic_data.append({
                "text": f"### Human: {q}\n\n### Assistant: {a}\n\n### End"
            })
    
    # Add more company-specific questions with expanded variations
    company_questions = [
        ("Why should we hire you?", "I bring a unique combination of technical expertise and leadership experience. My track record of delivering complex projects on time, mentoring team members, and driving technical decisions makes me a strong fit. I'm passionate about your mission and excited to contribute to your continued growth."),
        ("What interests you about this role?", "I'm excited about the technical challenges and the opportunity to work on systems at scale. The role aligns perfectly with my experience in distributed systems and my goal to drive architectural decisions. I'm also interested in mentoring junior engineers and building high-performing teams."),
        ("What's your ideal work environment?", "I thrive in collaborative environments that value learning and innovation. I appreciate clear communication, constructive feedback, and the autonomy to make technical decisions. I also value work-life balance and opportunities for professional growth."),
        ("How do you handle disagreements with management?", "I approach disagreements with data and clear reasoning. I present alternative solutions with their trade-offs and business impact. If management has different priorities, I align with their decision while documenting my concerns. Open communication and mutual respect are key."),
        ("Where do you see yourself in 5 years?", "I see myself in a technical leadership role where I can influence architectural decisions and mentor other engineers. I want to continue growing my expertise in distributed systems and contribute to building products that have significant impact on users and business."),
        ("What's your biggest weakness?", "I sometimes spend too much time perfecting code architecture when simpler solutions would work. I've learned to balance technical excellence with delivery timelines by setting time limits for design iterations and seeking feedback from colleagues earlier in the process."),
        ("Why are you leaving your current job?", "I'm looking for new challenges and growth opportunities. While I've learned a lot in my current role, I'm ready to take on more complex technical problems and leadership responsibilities. This role offers the perfect opportunity to expand my skills and make a greater impact."),
        ("How do you handle stress and pressure?", "I handle stress by staying organized and prioritizing tasks based on impact. I communicate proactively with stakeholders about timelines and potential issues. I also make sure to take breaks and maintain work-life balance to avoid burnout. Regular exercise and hobbies help me manage stress effectively.")
    ]
    
    for q, a in company_questions:
        for _ in range(35):  # Much more company-specific questions
            synthetic_data.append({
                "text": f"### Human: {q}\n\n### Assistant: {a}\n\n### End"
            })
    
    logger.info(f"Generated {len(synthetic_data)} comprehensive interview examples")
    
    random.shuffle(synthetic_data)
    
    dataset = Dataset.from_list(synthetic_data)
    split_dataset = dataset.train_test_split(test_size=0.08, seed=42)  # Even smaller test set for more training data
    
    return split_dataset['train'], split_dataset['test']

def train_interview_model_focused(output_dir, optimized=False):
    """Focused training specifically for interview model with comprehensive data and slower learning"""
    set_random_seeds(42)
    
    MODEL_ID = "gpt2"
    
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if "/kaggle/working" in os.getcwd():
        CACHE_DIR = "/kaggle/working/cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

    # Use comprehensive data generation
    train_dataset, eval_dataset = prepare_interview_data_comprehensive("synthetic")
    
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
        # Extremely conservative training configuration to prevent fast overfitting
        lora_config = get_lora_config(r=32, lora_alpha=64, lora_dropout=0.2)  # Smaller capacity, more regularization
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Minimum batch size
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,  # Very large accumulation for stable gradients
            learning_rate=5e-6,  # Much slower learning rate 
            num_train_epochs=25,  # Many more epochs for slow learning
            logging_steps=5,  # Very frequent logging
            eval_strategy="steps",
            eval_steps=200,  # Less frequent evaluation to allow more training
            save_strategy="steps",
            save_steps=400,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=True,
            warmup_steps=1000,  # Very long warmup for extremely gradual learning
            lr_scheduler_type="polynomial",  # Controlled decay
            optim="adamw_torch",
            weight_decay=0.02,  # More regularization
            max_grad_norm=0.1,  # Very strict gradient clipping
            seed=42,
            gradient_checkpointing=False,
            label_smoothing_factor=0.0,
            dataloader_drop_last=True,
            remove_unused_columns=True,
            prediction_loss_only=True,
            dataloader_num_workers=0,
            save_safetensors=True,
            disable_tqdm=False,
            # Additional slowdown parameters
            eval_accumulation_steps=8,  # Slower evaluation
            save_on_each_node=False,
        )
        
        callbacks = [
            ImprovedTargetLossCallback(target_loss=1.5),  # More conservative target
            EarlyStoppingCallback(early_stopping_patience=25)  # Much more patience
        ]
    else:
        lora_config = get_lora_config(r=16, lora_alpha=32, lora_dropout=0.15)  # Even smaller for standard mode
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,  # Slower standard rate
            num_train_epochs=15,  # More epochs
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            report_to="none",
            fp16=True,
            warmup_ratio=0.2,  # Longer warmup
            seed=42,
            weight_decay=0.01
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
        max_seq_length=384,
        packing=False,
        callbacks=callbacks,
    )
    
    logger.info(f"Starting comprehensive interview model training with {len(train_dataset)} examples")
    logger.info("This will take longer but should achieve better results")
    
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Interview model fine-tuning complete. Model saved to {output_dir}")

# --- Main Execution ---

def main():
    if not IMPORTS_SUCCESSFUL:
        logger.error("Required dependencies could not be imported")
        return
    
    parser = argparse.ArgumentParser(description="Fine-tune a model for a specific task.")
    parser.add_argument("--model_type", type=str, required=True, choices=["cover_letter", "cover_letter_ultra", "interview", "interview_focused"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--optimized", action="store_true", help="Optimized mode for better performance")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    
    if hf_token:
        logger.info("Logging in to Hugging Face Hub")
        try:
            login(token=hf_token)
            logger.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logger.error(f"Failed to authenticate: {e}")

    if args.model_type == "cover_letter_ultra":
        logger.info("ULTRA-OPTIMIZED mode: Maximum performance training for lowest loss")
    elif args.optimized:
        logger.info("Optimized mode: Enhanced training for better performance")
    else:
        logger.info("Standard mode: Quick training")

    if args.model_type == "cover_letter":
        logger.info("Starting Cover Letter Model Fine-Tuning")
        train_cover_letter_model(args.output_dir, optimized=args.optimized)
    elif args.model_type == "cover_letter_ultra":
        logger.info("Starting ULTRA-OPTIMIZED Cover Letter Model Fine-Tuning")
        train_ultra_optimized_cover_letter_model(args.output_dir)
    elif args.model_type == "interview":
        logger.info("Starting Interview Model Fine-Tuning")
        train_interview_model_focused(args.output_dir, optimized=args.optimized)
    elif args.model_type == "interview_focused":
        logger.info("Starting Focused Interview Model Fine-Tuning")
        train_interview_model_focused(args.output_dir, optimized=args.optimized)

if __name__ == "__main__":
    main() 