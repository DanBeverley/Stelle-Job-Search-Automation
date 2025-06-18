import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
from .. import schemas
from ..models.db import user as user_model

# --- Singleton Class for Cover Letter Generation Model ---

class CoverLetterGenerator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CoverLetterGenerator, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.model_loaded = False
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        if self.model_loaded:
            print("Model already loaded.")
            return

        # Prevent re-initialization in case of multi-threading issues during first load
        if hasattr(self, '_loading') and self._loading:
            return
        self._loading = True

        try:
            MODEL_ID = "google/gemma-2b-it"
            # Assuming the app runs from the project root 'Stelle-Job-Search-Automation'
            ADAPTER_PATH = "ai_job_search_app/final_model"
            CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

            if not os.path.exists(ADAPTER_PATH):
                raise FileNotFoundError(f"Adapter path not found: {ADAPTER_PATH}. Ensure the model is in the correct directory.")

            print("Configuring quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            print(f"Loading tokenizer for model: {MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Loading base model: {MODEL_ID}")
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=CACHE_DIR
            )

            print(f"Loading and merging LoRA adapter from: {ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            # self.model = self.model.merge_and_unload() # Optional: merge if you don't need to train anymore
            
            self.model.eval() # Set model to evaluation mode

            self.model_loaded = True
            print("--- Model loaded successfully ---")

        except Exception as e:
            print(f"--- FATAL: Error loading model: {e} ---")
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
        finally:
            self._loading = False
            
    def generate(self, prompt: str) -> str:
        if not self.model_loaded or not self.tokenizer:
            error_message = "Error: Model is not loaded. Please check the server logs for details."
            print(error_message)
            return error_message

        # The prompt already includes the '<start_of_turn>model' token
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)

        # Generate response
        with torch.no_grad():
            # max_new_tokens can be adjusted. 1024 is safer for long cover letters.
            outputs = self.model.generate(**input_ids, max_new_tokens=1024, do_sample=True, top_p=0.95, temperature=0.8)

        # Decode only the newly generated tokens, skipping the prompt
        generated_text = self.tokenizer.decode(outputs[0][len(input_ids['input_ids'][0]):], skip_special_tokens=True)
        
        # Clean up potential artifacts
        end_of_turn_token = self.tokenizer.eos_token # Or "<end_of_turn>" if it's a special token
        if end_of_turn_token in generated_text:
            generated_text = generated_text.split(end_of_turn_token)[0]

        return generated_text.strip()

# --- Service Function ---

# Instantiate the generator when the module is first imported.
# FastAPI will manage its lifecycle.
cover_letter_generator = CoverLetterGenerator()

def generate_cover_letter_with_finetuned_model(user: user_model.User, request: schemas.CoverLetterRequest) -> schemas.CoverLetterResponse:
    """
    Generates a cover letter using the fine-tuned Gemma model.
    """
    if not cover_letter_generator.model_loaded:
         raise RuntimeError("Cover letter generation model could not be loaded. Please check server logs.")

    if not user.parsed_cv_data:
        # This check should ideally be in the API layer to return a proper HTTP error
        raise ValueError("No CV data found for this user. Please upload and parse a CV first.")

    cv_skills = ", ".join(user.parsed_cv_data.get('skills', ['Not available']))
    experiences = user.parsed_cv_data.get('experiences', [])
    cv_experience = "\n".join([f"- {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}" for exp in experiences]) if experiences else "Not available"

    # This prompt structure MUST match the one used in `cover_letter_finetuning.py`
    prompt = f"""<start_of_turn>user
Generate a professional cover letter based on the following job details and candidate information.

**Job Title:**
{request.job_title}

**Hiring Company:**
{request.company}

**Preferred Qualifications:**
{request.job_description}

**Candidate's Past Experience:**
{cv_experience}

**Candidate's Skills:**
{cv_skills}<end_of_turn>
<start_of_turn>model
"""

    print("--- Generating Cover Letter ---")
    generated_text = cover_letter_generator.generate(prompt)
    print("--- Cover Letter Generated ---")

    return schemas.CoverLetterResponse(
        cover_letter_text=generated_text,
        prompt_used=prompt
    ) 