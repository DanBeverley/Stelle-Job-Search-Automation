import torch
import os
import argparse

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

# --- Utility Functions ---

def get_quantization_config():
    """Returns the BitsAndBytes configuration for 4-bit quantization (QLoRA)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # More memory efficient
        bnb_4bit_quant_storage=torch.uint8  # Reduce storage requirements
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

# --- Cover Letter Model ---

def prepare_cover_letter_data(dataset_name, cache_dir):
    """Loads and prepares the cover letter dataset, splitting it into training and validation sets."""
    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=42)
    # 90% for training, 10% for validation
    split_dataset = dataset.train_test_split(test_size=0.1)
    return split_dataset['train'], split_dataset['test']

def format_cover_letter_prompt(data_point):
    """Formats the prompt for the cover letter model."""
    return f"""<start_of_turn>user
Generate a professional cover letter based on the following job details and candidate information.
**Job Title:** {data_point.get('Job Title', 'N/A')}
**Hiring Company:** {data_point.get('Hiring Company', 'N/A')}
**Preferred Qualifications:** {data_point.get('Preferred Qualifications', 'N/A')}
**Candidate's Past Experience:** {data_point.get('Past Working Experience', 'N/A')}
**Candidate's Skills:** {data_point.get('Skillsets', 'N/A')}<end_of_turn>
<start_of_turn>model
{data_point.get('Cover Letter', 'N/A')}<end_of_turn>"""

def train_cover_letter_model(output_dir):
    """Fine-tunes the cover letter generation model with state-of-the-art optimization techniques."""
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Smaller model to fit in Kaggle's 16GB GPU
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    train_dataset, eval_dataset = prepare_cover_letter_data("ShashiVish/cover-letter-dataset", CACHE_DIR)
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", cache_dir=CACHE_DIR)

    # Qwen2.5 specific target modules for LoRA
    lora_config = get_lora_config(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Keep small for memory
        per_device_eval_batch_size=1,   # Small eval batch size
        gradient_accumulation_steps=8,  # Maintain effective batch size
        learning_rate=1e-4,
        num_train_epochs=2,  # Reduced epochs to prevent OOM
        logging_steps=10,
        eval_strategy="steps",  # Updated parameter name
        eval_steps=50,
        save_strategy="steps",
        save_total_limit=1,  # Keep only 1 checkpoint to save memory
        load_best_model_at_end=False,  # Disable to save memory
        report_to="none",
        fp16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,  # Reduce CPU memory usage
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,  # Trade compute for memory
        remove_unused_columns=True,   # Remove unused data columns
        max_grad_norm=1.0  # Gradient clipping
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="Cover Letter",
        tokenizer=tokenizer,
        packing=True,
        max_seq_length=512,  # Reduced to save memory
        formatting_func=format_cover_letter_prompt,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Cover letter model fine-tuning complete. Model saved to {output_dir}")

# --- Interview Question Model ---

def prepare_interview_data(dataset_name):
    """Creates a comprehensive synthetic interview dataset for training."""
    # Create much more synthetic interview data to prevent packing errors
    synthetic_data = []
    
    job_descriptions = [
        "Software Engineer position requiring Python, JavaScript, and React experience with 3+ years experience",
        "Senior Data Scientist role focusing on machine learning, deep learning, and statistical analysis",
        "Product Manager position requiring stakeholder management, strategic planning, and agile methodologies",
        "DevOps Engineer role with AWS, Docker, Kubernetes, and CI/CD pipeline experience",
        "UX Designer position requiring user research, prototyping, and design thinking skills",
        "Full Stack Developer role with Node.js, Python, and database design experience",
        "Machine Learning Engineer position requiring MLOps, model deployment, and cloud platforms",
        "Frontend Developer role with React, Vue.js, and modern JavaScript frameworks",
        "Backend Developer position requiring API design, microservices, and database optimization",
        "Cloud Architect role with AWS/Azure expertise and infrastructure as code experience"
    ]
    
    interview_questions = [
        "Tell me about a challenging project you worked on and how you overcame obstacles.",
        "How do you approach problem-solving when faced with a complex technical issue?",
        "Describe a time when you had to work with a difficult team member.",
        "What motivates you in your work and how do you stay current with industry trends?",
        "How do you prioritize tasks when managing multiple projects simultaneously?",
        "Explain a situation where you had to learn a new technology quickly.",
        "Describe your experience with agile development methodologies.",
        "How do you handle feedback and criticism of your work?",
        "Tell me about a time you had to make a difficult technical decision.",
        "What's your approach to code review and maintaining code quality?",
        "How do you ensure your solutions are scalable and maintainable?",
        "Describe a time when you had to debug a complex production issue.",
        "How do you stay organized when working on multiple features?",
        "Tell me about your experience with testing and quality assurance.",
        "How do you approach documentation and knowledge sharing?"
    ]
    
    # Generate many more combinations to ensure sufficient data
    for i, job_desc in enumerate(job_descriptions):
        for j, question in enumerate(interview_questions):
            synthetic_data.append({
                "text": f"""### INSTRUCTION:
Generate an interview question for the following candidate and job role.
### JOB DESCRIPTION:
{job_desc}
### CANDIDATE CV:
Experienced professional with relevant skills and background in the field.
### INTERVIEW QUESTION:
{question}"""
            })
    
    print(f"Generated {len(synthetic_data)} synthetic interview examples")
    
    # Create dataset with more examples
    dataset = Dataset.from_list(synthetic_data)
    dataset = dataset.shuffle(seed=42)
    # Use 80/20 split to ensure both splits have enough examples
    split_dataset = dataset.train_test_split(test_size=0.2)
    
    print(f"Training examples: {len(split_dataset['train'])}")
    print(f"Evaluation examples: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']

def train_interview_model(output_dir):
    """Fine-tunes the interview question generation model with state-of-the-art optimization techniques."""
    MODEL_ID = "microsoft/DialoGPT-medium"  # Non-gated alternative that works well

    train_dataset, eval_dataset = prepare_interview_data("synthetic")
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model.config.use_cache = False

    # DialoGPT specific target modules for LoRA
    lora_config = get_lora_config(r=16, lora_alpha=32, target_modules=["c_attn", "c_proj"], lora_dropout=0.05)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Can use larger batch for smaller model
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        max_steps=200,  # Reasonable training steps
        eval_strategy="steps",  # Updated parameter name
        eval_steps=30,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,  # Save memory
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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=256,  # Smaller for memory efficiency
        tokenizer=tokenizer,
        packing=True
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Interview model fine-tuning complete. Model saved to {output_dir}")


# --- Main Execution ---

def main():
    # Check if imports were successful
    if not IMPORTS_SUCCESSFUL:
        print("ERROR: Required dependencies could not be imported.")
        print("Please install compatible versions of the required packages:")
        print("pip install torch>=2.1.0,<2.5.0 transformers>=4.36.0,<4.46.0 accelerate>=0.25.0,<0.35.0")
        return
    
    parser = argparse.ArgumentParser(description="Fine-tune a model for a specific task.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["cover_letter", "interview"],
        help="The type of model to fine-tune."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the fine-tuned model."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face token for authentication (overrides environment variables)"
    )
    args = parser.parse_args()

    # --- Hugging Face Login ---
    # Check for the Hugging Face token - prioritize command line argument
    hf_token = args.hf_token
    if not hf_token:
        # Try environment variables as fallback
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            # Try alternative environment variable names that Kaggle might use
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        try:
            login(token=hf_token)
            print("Successfully authenticated with Hugging Face!")
        except Exception as e:
            print(f"Failed to authenticate: {e}")
    else:
        print("WARNING: No Hugging Face token found. Pre-trained model downloads may fail.")
        print("Please provide HF_TOKEN via --hf_token argument or set in Kaggle Secrets.")


    if args.model_type == "cover_letter":
        print("--- Starting Cover Letter Model Fine-Tuning ---")
        train_cover_letter_model(args.output_dir)
    elif args.model_type == "interview":
        print("--- Starting Interview Model Fine-Tuning ---")
        train_interview_model(args.output_dir)
    else:
        print("Invalid model type specified.")

if __name__ == "__main__":
    main() 