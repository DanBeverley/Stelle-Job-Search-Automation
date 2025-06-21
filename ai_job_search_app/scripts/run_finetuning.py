import torch
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
import os
import argparse

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
    """Fine-tunes the cover letter generation model with optimization techniques."""
    MODEL_ID = "google/gemma-2b-it"
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    train_dataset, eval_dataset = prepare_cover_letter_data("ShashiVish/cover-letter-dataset", CACHE_DIR)
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", cache_dir=CACHE_DIR)

    lora_config = get_lora_config(r=8, lora_alpha=32, target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="steps", # Evaluate at each logging step
        eval_steps=50,               # Evaluation frequency
        save_strategy="steps",       # Save model based on evaluation
        save_total_limit=2,          # Only keep the best and the latest model
        load_best_model_at_end=True, # Load the best model at the end of training
        report_to="none",
        fp16=True,
        weight_decay=0.01,           # L2 Regularization
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
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
        max_seq_length=1024,
        formatting_func=format_cover_letter_prompt,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/final_model")
    print(f"Cover letter model fine-tuning complete. Model saved to {output_dir}/final_model")

# --- Interview Question Model ---

def prepare_interview_data(dataset_name):
    """Loads and prepares the interview question dataset, splitting it."""
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=42)
    # Using a subset for demo and splitting it
    split_dataset = dataset.select(range(2000)).train_test_split(test_size=0.2)
    return split_dataset['train'].map(lambda x: {"text": format_interview_prompt(x)}), \
           split_dataset['test'].map(lambda x: {"text": format_interview_prompt(x)})


def format_interview_prompt(data_point):
    """Formats the prompt for the interview question model."""
    job_title = data_point['category']
    question = data_point['question']
    skills = ["teamwork", "problem-solving", job_title.lower().replace("_", " ")]
    experience = "previous projects in a similar domain"
    job_description = f"Seeking a {job_title} with experience in {', '.join(skills)}."
    cv_summary = f"Candidate has skills in {', '.join(skills)} and experience in {experience}."

    return f"""### INSTRUCTION:
Generate an interview question for the following candidate and job role.
### JOB DESCRIPTION:
{job_description}
### CANDIDATE CV:
{cv_summary}
### INTERVIEW QUESTION:
{question}"""

def train_interview_model(output_dir):
    """Fine-tunes the interview question generation model with optimization techniques."""
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

    train_dataset, eval_dataset = prepare_interview_data("jacob-hugging-face/job-interview-question-dataset")
    
    bnb_config = get_quantization_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    lora_config = get_lora_config(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2, # Reduced for stability
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200, # Increased for meaningful training
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        weight_decay=0.01 # L2 Regularization
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Interview model fine-tuning complete. Model saved to {output_dir}")


# --- Main Execution ---

def main():
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