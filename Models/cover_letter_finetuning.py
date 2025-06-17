import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- Model and Tokenizer Configuration ---
MODEL_ID = "google/gemma-2b-it"
# Use a local directory to cache the model, avoiding re-downloads
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

def generate_prompt(data_point):
    """
    Generates a structured prompt for fine-tuning the model.
    """
    # The prompt now needs to be a single string for the SFTTrainer's text_field
    return f"""<start_of_turn>user
Generate a professional cover letter based on the following job details and candidate information.

**Job Title:**
{data_point.get('Job Title', 'N/A')}

**Hiring Company:**
{data_point.get('Hiring Company', 'N/A')}

**Preferred Qualifications:**
{data_point.get('Preferred Qualifications', 'N/A')}

**Candidate's Past Experience:**
{data_point.get('Past Working Experience', 'N/A')}

**Candidate's Skills:**
{data_point.get('Skillsets', 'N/A')}<end_of_turn>
<start_of_turn>model
{data_point.get('Cover Letter', 'N/A')}<end_of_turn>"""

def main():
    # --- 1. Load and Process Dataset ---
    print("Loading dataset...")
    dataset = load_dataset("ShashiVish/cover-letter-dataset", split="train", cache_dir=CACHE_DIR)
    dataset = dataset.shuffle(seed=42)

    # --- 2. Configure Quantization (for memory efficiency) ---
    print("Configuring quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- 3. Load Tokenizer and Model ---
    print(f"Loading tokenizer for model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    # Gemma's tokenizer doesn't have a pad_token, so we use the eos_token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0}, # Auto-detects and uses the GPU
        cache_dir=CACHE_DIR
    )

    # --- 4. Configure LoRA (Parameter-Efficient Fine-Tuning) ---
    print("Configuring LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    # --- 5. Configure Training Arguments ---
    output_dir = "./Models/gemma-cover-letter-generator"
    print(f"Output directory set to: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2, # Reduce if you encounter memory issues
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,          # Using 3 epochs for a good balance of training
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",            # Can be "tensorboard" or "wandb" if you have them configured
        fp16=True,                   # Use mixed precision training
    )

    # --- 6. Initialize and Start the Trainer ---
    print("Initializing Supervised Fine-tuning (SFT) Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="Cover Letter", # This should be the field with the full text
        tokenizer=tokenizer,
        packing=True, # Packs multiple short examples into one sequence for efficiency
        max_seq_length=1024, # Adjust based on VRAM and average sample length
        formatting_func=generate_prompt, # Use our custom prompt formatter
    )
    
    print("\n--- Starting model fine-tuning... ---")
    trainer.train()
    print("--- Fine-tuning complete. ---")

    # --- 7. Save the Fine-tuned Model ---
    final_model_path = f"{output_dir}/final"
    print(f"Saving final model adapter to {final_model_path}...")
    trainer.save_model(final_model_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main() 