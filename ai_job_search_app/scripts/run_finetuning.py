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
        bnb_4bit_compute_dtype=torch.bfloat16
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
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # State-of-the-art model, no access required
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
        per_device_train_batch_size=1,  # Reduced for 7B model
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        learning_rate=1e-4,  # Slightly lower for stability
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=True,
        dataloader_pin_memory=False,
        weight_decay=0.01,
        warmup_ratio=0.1,  # Warmup for better training
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        optim="adamw_torch"  # Optimized AdamW
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
        max_seq_length=1024,  # Restored for better quality
        formatting_func=format_cover_letter_prompt,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Cover letter model fine-tuning complete. Model saved to {output_dir}")

# --- Interview Question Model ---

def prepare_interview_data(dataset_name):
    """Creates a simple synthetic interview dataset for training."""
    # Create synthetic interview data since the original dataset doesn't exist
    synthetic_data = []
    
    job_descriptions = [
        "Software Engineer position requiring Python, JavaScript, and React experience",
        "Data Scientist role focusing on machine learning and statistical analysis",
        "Product Manager position requiring stakeholder management and strategic planning",
        "DevOps Engineer role with AWS, Docker, and Kubernetes experience",
        "UX Designer position requiring user research and prototyping skills"
    ]
    
    interview_questions = [
        "Tell me about a challenging project you worked on and how you overcame obstacles.",
        "How do you approach problem-solving when faced with a complex technical issue?",
        "Describe a time when you had to work with a difficult team member.",
        "What motivates you in your work and how do you stay current with industry trends?",
        "How do you prioritize tasks when managing multiple projects simultaneously?"
    ]
    
    for i, job_desc in enumerate(job_descriptions):
        for j, question in enumerate(interview_questions):
            synthetic_data.append({
                "text": f"### JOB DESCRIPTION:\n{job_desc}\n### INTERVIEW QUESTION:\n{question}"
            })
    
    # Create dataset
    dataset = Dataset.from_list(synthetic_data)
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=0.2)
    return split_dataset['train'], split_dataset['test']


def format_interview_prompt(data_point):
    """Formats the prompt for the interview question model using the new dataset structure."""
    # The dataset contains a 'prompt' with the job description and a 'completion' with the question.
    job_description = data_point['prompt'] # This field contains the job description
    question = data_point['completion']   # This field contains the corresponding question

    # We can create a simple CV summary as the dataset doesn't provide one.
    cv_summary = "Candidate with relevant skills and experience."

    # The final prompt structure should still match what the model expects
    formatted_prompt = f"""### INSTRUCTION:
Generate an interview question for the following candidate and job role.
### JOB DESCRIPTION:
{job_description}
### CANDIDATE CV:
{cv_summary}
### INTERVIEW QUESTION:
{question}"""
    return {"text": formatted_prompt}

def train_interview_model(output_dir):
    """Fine-tunes the interview question generation model with state-of-the-art optimization techniques."""
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Latest Mistral model

    train_dataset, eval_dataset = prepare_interview_data("synthetic")
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model.config.use_cache = False

    # Mistral specific target modules for LoRA
    lora_config = get_lora_config(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # Reduced for 7B model
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        learning_rate=1e-4,  # Slightly lower for stability
        logging_steps=10,
        max_steps=300,  # Increased for meaningful training
        evaluation_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=True,
        dataloader_pin_memory=False,
        weight_decay=0.01,
        warmup_ratio=0.1,  # Warmup for better training
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        optim="adamw_torch"  # Optimized AdamW
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,  # Restored for better quality
        tokenizer=tokenizer,
        packing=True  # Enable packing for efficiency
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
    args = parser.parse_args()

    # --- Hugging Face Login ---
    # Check for the Hugging Face token as an environment variable.
    # On Kaggle, this should be set in the "Secrets" section of your notebook.
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("WARNING: No Hugging Face token found. Pre-trained model downloads may fail.")


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