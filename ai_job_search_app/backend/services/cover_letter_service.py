import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import logging
from .. import schemas
from ..models.db import user as user_model

logger = logging.getLogger(__name__)

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
            logger.info("Model already loaded.")
            return

        # Prevent re-initialization in case of multi-threading issues during first load
        if hasattr(self, '_loading') and self._loading:
            return
        self._loading = True

        try:
            MODEL_ID = "gpt2"  # Updated to use GPT-2 base model
            # Path to the cover letter checkpoint
            ADAPTER_PATH = "ai_job_search_app/final_model/checkpoint-420"
            CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

            if not os.path.exists(ADAPTER_PATH):
                raise FileNotFoundError(f"Adapter path not found: {ADAPTER_PATH}. Ensure the model is in the correct directory.")

            logger.info("Loading tokenizer for model: %s", MODEL_ID)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading base model: %s", MODEL_ID)
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            logger.info("Loading and merging LoRA adapter from: %s", ADAPTER_PATH)
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            
            self.model.eval() # Set model to evaluation mode

            self.model_loaded = True
            logger.info("--- Cover letter model loaded successfully ---")

        except Exception as e:
            logger.error("--- FATAL: Error loading model: %s ---", e)
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
        finally:
            self._loading = False
            
    def generate(self, prompt: str) -> str:
        if not self.model_loaded or not self.tokenizer:
            error_message = "Error: Model is not loaded. Please check the server logs for details."
            logger.error(error_message)
            return error_message

        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to the appropriate device
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            self.model = self.model.cuda()

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 500,  # Generate up to 500 new tokens
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens, skipping the prompt
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the generated text
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

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

    # Updated prompt structure to match the GPT-2 training format
    prompt = f"""### Instruction:
Write a professional cover letter for the {request.job_title} position at {request.company}.

Job Description: {request.job_description}
Candidate Experience: {cv_experience}
Candidate Skills: {cv_skills}

### Response:"""

    logger.info("--- Generating Cover Letter ---")
    generated_text = cover_letter_generator.generate(prompt)
    logger.info("--- Cover Letter Generated ---")

    return schemas.CoverLetterResponse(
        cover_letter_text=generated_text,
        prompt_used=prompt
    ) 