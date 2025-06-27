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

def create_synthetic_cover_letters(n_samples=2000):
    set_random_seeds(42)
    
    job_titles = [
        "Software Engineer", "Senior Software Engineer", "Data Scientist", "Product Manager", 
        "DevOps Engineer", "Full Stack Developer", "Backend Developer", "Frontend Developer",
        "Machine Learning Engineer", "Technical Lead", "Engineering Manager", "QA Engineer"
    ]
    
    companies = [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Uber", "Airbnb",
        "Tesla", "SpaceX", "Stripe", "Shopify", "Adobe", "Salesforce", "Oracle", "IBM"
    ]
    
    templates = [
        """Dear Hiring Manager,

I am writing to express my strong interest in the {job_title} position at {company}. With {years} years of experience in {domain}, I am excited about the opportunity to contribute to your innovative team.

My expertise in {tech_stack} has enabled me to {achievement}. I am particularly drawn to {company}'s {company_strength} and believe my background in {specialty} would be valuable for your {department} team.

In my previous role, I {previous_work} which resulted in {outcome}. I am passionate about {passion} and continuously stay updated with {learning_area}.

Thank you for considering my application. I look forward to discussing how I can contribute to {company}'s continued success.

Best regards,
[Your Name]""",

        """Dear {company} Team,

I am excited to apply for the {job_title} role. As a {experience_level} professional with {years} years in {field}, I have developed strong skills in {skills} that align perfectly with your requirements.

Throughout my career, I have {career_highlight} and {project_work}. My experience with {tech_stack} and {methodology} has prepared me to tackle the challenges facing {company}'s {team} team.

I am particularly interested in {company} because of {reason}. Your commitment to {value} resonates with my professional values and career goals.

I would welcome the opportunity to discuss how my {strength1} and {strength2} can contribute to {company}'s mission.

Sincerely,
[Your Name]""",

        """Hello {company} Hiring Team,

I am writing to apply for the {job_title} position. With a background in {background} and {years} years of hands-on experience, I am confident I can make a meaningful contribution to your team.

My technical expertise includes {tech_stack}, and I have successfully {success_story}. I am passionate about {tech_interest} and have experience in {additional_skills}.

What excites me most about {company} is {excitement_reason}. I am eager to bring my skills in {key_skills} to help {company} {company_goal}.

Thank you for your time and consideration. I look forward to the opportunity to discuss my qualifications further.

Best,
[Your Name]"""
    ]
    
    tech_stacks = [
        "Python, Django, PostgreSQL", "JavaScript, React, Node.js", "Java, Spring Boot, MySQL",
        "Go, Docker, Kubernetes", "C#, .NET, Azure", "Ruby on Rails, Redis", "Swift, iOS Development",
        "Kotlin, Android Development", "Vue.js, TypeScript", "Angular, RxJS", "React Native, Firebase",
        "PHP, Laravel, MongoDB", "Scala, Spark, Kafka", "Rust, WebAssembly", "Flutter, Dart"
    ]
    
    domains = ["software development", "data science", "machine learning", "web development", "mobile development", "cloud computing"]
    achievements = ["deliver scalable solutions", "optimize system performance", "lead cross-functional teams", "implement best practices"]
    company_strengths = ["innovative culture", "technical excellence", "market leadership", "commitment to quality"]
    
    data = []
    for i in range(n_samples):
        template = random.choice(templates)
        job_title = random.choice(job_titles)
        company = random.choice(companies)
        
        cover_letter = template.format(
            job_title=job_title,
            company=company,
            years=random.randint(2, 8),
            domain=random.choice(domains),
            tech_stack=random.choice(tech_stacks),
            achievement=random.choice(achievements),
            company_strength=random.choice(company_strengths),
            specialty="backend systems" if "Backend" in job_title else "full-stack development",
            department="engineering",
            previous_work="led the development of microservices architecture",
            outcome="40% improvement in system performance",
            passion="building scalable applications",
            learning_area="emerging technologies",
            experience_level="experienced",
            field="software engineering",
            skills="problem-solving and system design",
            career_highlight="successfully delivered multiple high-impact projects",
            project_work="collaborated with product teams to define technical requirements",
            methodology="agile development practices",
            team="product",
            reason="your reputation for innovation",
            value="technical excellence",
            strength1="analytical thinking",
            strength2="collaborative approach",
            background="computer science",
            success_story="reduced deployment time by 60%",
            tech_interest="cloud-native technologies",
            additional_skills="DevOps and CI/CD",
            excitement_reason="the opportunity to work on cutting-edge technology",
            key_skills="software architecture and team leadership",
            company_goal="achieve its ambitious growth targets"
        )
        
        prompt = f"Write a professional cover letter for a {job_title} position at {company}."
        
        data.append({
            "text": f"### Human: {prompt}\n\n### Assistant: {cover_letter}\n\n### End"
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
    logger.info("Using synthetic cover letter data for reliable training")
    return create_synthetic_cover_letters()

def create_synthetic_interview_data(n_samples=2000):
    set_random_seeds(42)
    
    questions = [
        "Tell me about yourself and your background",
        "What are your key strengths and how do they apply to this role?",
        "Why do you want to work for our company?",
        "Describe a challenging project you've worked on recently",
        "How do you handle stress and tight deadlines?",
        "Where do you see yourself in 5 years?",
        "What motivates you in your work?",
        "How do you stay updated with new technologies?",
        "Describe a time when you had to work with a difficult team member",
        "What's your biggest professional achievement?",
        "How do you approach problem-solving?",
        "What are your salary expectations?",
        "Why are you looking to leave your current position?",
        "How do you prioritize your work when you have multiple deadlines?",
        "What programming languages are you most comfortable with?",
        "Describe your experience with agile development methodologies",
        "How do you ensure code quality in your projects?",
        "What's the most complex system you've designed?",
        "How do you handle feedback and criticism?",
        "What questions do you have for us?"
    ]
    
    response_templates = [
        "I'm a {role} with {years} years of experience in {tech}. I've worked on various projects including {project_type}, and I'm passionate about {passion}. I have strong skills in {skills} and enjoy {activity}.",
        "My key strengths include {strength1}, {strength2}, and {strength3}. These help me {benefit} and deliver {outcome}. For example, {example}.",
        "I'm excited about this opportunity because {reason1}. Your company's {company_aspect} aligns with my {value}. I believe I can contribute by {contribution}.",
        "Recently, I worked on {project} which required {tech_stack}. The main challenge was {challenge}, which I solved by {solution}. The result was {outcome}.",
        "I handle stress by {method1} and {method2}. When facing tight deadlines, I {approach} and {strategy}. This helps me maintain {quality} while meeting requirements.",
        "In 5 years, I see myself as {future_role} where I can {future_goal}. I plan to develop my skills in {skill_area} and {growth_area}.",
        "I'm motivated by {motivation1} and {motivation2}. I find satisfaction in {satisfaction} and enjoy {enjoyment}.",
        "I stay updated through {method1}, {method2}, and {method3}. I regularly {activity} and participate in {community}.",
        "I approach difficult situations by {approach1} and {approach2}. Communication is key, so I {communication_strategy}.",
        "My biggest achievement was {achievement} where I {action} and achieved {result}. This demonstrated my ability to {skill}."
    ]
    
    # Define replacement values
    roles = ["Software Engineer", "Full Stack Developer", "Backend Developer", "Frontend Developer", "DevOps Engineer", "Data Scientist"]
    years_exp = ["2", "3", "4", "5", "6", "7", "8"]
    technologies = ["Python and Django", "JavaScript and React", "Java and Spring", "Node.js and MongoDB", "AWS and Docker"]
    project_types = ["web applications", "microservices", "data pipelines", "mobile apps", "automation tools"]
    passions = ["clean code", "user experience", "system architecture", "performance optimization", "continuous learning"]
    skills = ["problem-solving", "team collaboration", "technical leadership", "code review", "system design"]
    
    data = []
    for i in range(n_samples):
        question = random.choice(questions)
        template = random.choice(response_templates)
        
        # Fill template with random values
        response = template.format(
            role=random.choice(roles),
            years=random.choice(years_exp),
            tech=random.choice(technologies),
            project_type=random.choice(project_types),
            passion=random.choice(passions),
            skills=", ".join(random.sample(skills, 2)),
            activity="learning new technologies",
            strength1="analytical thinking",
            strength2="attention to detail", 
            strength3="effective communication",
            benefit="deliver high-quality solutions",
            outcome="successful project delivery",
            example="in my last project, these skills helped me identify and resolve critical bugs before deployment",
            reason1="the innovative projects and growth opportunities",
            company_aspect="commitment to technological excellence",
            value="professional development goals",
            contribution="leveraging my experience to drive technical innovation",
            project="a scalable e-commerce platform",
            tech_stack="microservices architecture with Docker and Kubernetes",
            challenge="handling high traffic loads during peak seasons",
            solution="implementing caching strategies and load balancing",
            method1="breaking down tasks into manageable chunks",
            method2="maintaining clear communication with stakeholders",
            method3="contributing to open-source projects",
            approach="prioritizing critical tasks",
            strategy="working closely with the team",
            quality="high code quality",
            future_role="a senior technical lead",
            future_goal="mentor junior developers and drive architectural decisions",
            skill_area="machine learning",
            growth_area="cloud technologies",
            motivation1="solving complex technical challenges",
            motivation2="seeing the impact of my work on users",
            satisfaction="building efficient, scalable solutions",
            enjoyment="collaborating with cross-functional teams",
            community="developer communities and forums",
            approach1="listening actively to understand their perspective",
            approach2="finding common ground and shared goals",
            communication_strategy="schedule one-on-one meetings to address concerns",
            achievement="leading the migration of our legacy system to a modern architecture",
            action="coordinated a team of 5 developers",
            result="50% improvement in system performance",
            skill="lead complex technical initiatives"
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

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return super().on_evaluate(args, state, control, logs=logs, **kwargs)
            
        current_loss = logs.get("eval_loss")
        if current_loss and current_loss < self.target_loss:
            logger.info(f"Target loss {self.target_loss} achieved! Current loss: {current_loss}")
            control.should_training_stop = True
        
        if current_loss and current_loss < self.best_loss:
            self.best_loss = current_loss
        
        return super().on_evaluate(args, state, control, logs=logs, **kwargs)

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
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        num_train_epochs=10,
        eval_strategy="steps",
        eval_steps=80,
        save_strategy="steps", 
        save_steps=80,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=300,
        logging_steps=20,
        lr_scheduler_type="cosine",
        weight_decay=0.01
    )
    
    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        formatting_func=lambda x: x["text"],
        callbacks=[EarlyStoppingWithTargetLoss(target_loss=0.9, patience=8)]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_interview_model(output_dir):
    logger.info("Training interview model with production settings")
    
    dataset = prepare_interview_data()
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"Interview training samples: {len(split_dataset['train'])}")
    logger.info(f"Interview eval samples: {len(split_dataset['test'])}")
    
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
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        num_train_epochs=10,
        eval_strategy="steps",
        eval_steps=60,
        save_strategy="steps",
        save_steps=60,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=300,
        logging_steps=10,
        lr_scheduler_type="cosine",
        weight_decay=0.01
    )
    
    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        formatting_func=lambda x: x["text"],
        callbacks=[EarlyStoppingWithTargetLoss(target_loss=0.9, patience=8)]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_salary_model(output_dir):
    logger.info("Training salary model using XGBoost (lightweight alternative)")
    
    # Create a simple salary prediction model using synthetic data
    try:
        import xgboost as xgb
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder
        import joblib
        
        # Generate synthetic salary data
        set_random_seeds(42)
        
        job_titles = ["Software Engineer", "Senior Software Engineer", "Lead Developer", "Engineering Manager", 
                     "Data Scientist", "DevOps Engineer", "Full Stack Developer", "Backend Developer", 
                     "Frontend Developer", "Mobile Developer"]
        
        experience_levels = ["Entry", "Mid", "Senior", "Lead"]
        locations = ["San Francisco", "New York", "Seattle", "Austin", "Remote"]
        
        # Create synthetic dataset
        data = []
        for _ in range(5000):
            title = random.choice(job_titles)
            exp = random.choice(experience_levels)
            loc = random.choice(locations)
            
            # Generate salary based on simple rules
            base_salary = 70000
            if "Senior" in title or "Lead" in title:
                base_salary += 30000
            if "Manager" in title:
                base_salary += 50000
            if exp == "Senior":
                base_salary += 20000
            elif exp == "Lead":
                base_salary += 40000
            if loc in ["San Francisco", "New York"]:
                base_salary += 20000
            elif loc == "Seattle":
                base_salary += 15000
            
            # Add some randomness
            salary = base_salary + random.randint(-15000, 25000)
            
            data.append({
                "job_title": title,
                "experience_level": exp,
                "location": loc,
                "salary": salary
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic salary records")
        
        # Prepare features
        vectorizer = TfidfVectorizer(max_features=100)
        title_features = vectorizer.fit_transform(df['job_title']).toarray()
        
        le_exp = LabelEncoder()
        exp_features = le_exp.fit_transform(df['experience_level']).reshape(-1, 1)
        
        le_loc = LabelEncoder()
        loc_features = le_loc.fit_transform(df['location']).reshape(-1, 1)
        
        # Combine features
        import numpy as np
        X = np.hstack([title_features, exp_features, loc_features])
        y = df['salary'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        logger.info("Training XGBoost salary prediction model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Training R²: {train_score:.3f}")
        logger.info(f"Test R²: {test_score:.3f}")
        
        # Save model and preprocessing components
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(model, os.path.join(output_dir, "salary_model.pkl"))
        joblib.dump(vectorizer, os.path.join(output_dir, "title_vectorizer.pkl"))
        joblib.dump(le_exp, os.path.join(output_dir, "experience_encoder.pkl"))
        joblib.dump(le_loc, os.path.join(output_dir, "location_encoder.pkl"))
        
        # Save model info
        model_info = {
            "model_type": "xgboost",
            "train_r2": train_score,
            "test_r2": test_score,
            "features": ["job_title", "experience_level", "location"],
            "target": "salary"
        }
        
        import json
        with open(os.path.join(output_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Salary model training completed successfully")
        logger.info(f"Model saved to: {output_dir}")
        
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