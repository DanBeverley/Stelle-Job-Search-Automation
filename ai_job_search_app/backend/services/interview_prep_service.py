import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class InterviewPrepGenerator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InterviewPrepGenerator, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.model_loaded = False
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        if self.model_loaded:
            logger.info("Interview model already loaded.")
            return

        try:
            MODEL_ID = "gpt2"
            # Path to the interview model (absolute path)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            ADAPTER_PATH = os.path.join(project_root, 'final_model', 'interview_model')
            CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

            if not os.path.exists(ADAPTER_PATH):
                raise FileNotFoundError(f"Adapter path not found: {ADAPTER_PATH}. Ensure the model is in the correct directory.")

            logger.info("Loading tokenizer for interview model: %s", MODEL_ID)
            self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading base model: %s", MODEL_ID)
            base_model = GPT2LMHeadModel.from_pretrained(
                MODEL_ID,
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            logger.info("Loading interview LoRA adapter from: %s", ADAPTER_PATH)
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            
            self.model.eval()
            self.model_loaded = True
            logger.info("--- Interview preparation model loaded successfully ---")

        except Exception as e:
            logger.error("--- FATAL: Error loading interview model: %s ---", e)
            self.model = None
            self.tokenizer = None
            self.model_loaded = False

    def generate_response(self, question: str) -> str:
        if not self.model_loaded or not self.tokenizer:
            return "Error: Interview model is not loaded."

        # Format prompt to match training format
        prompt = f"### Human: {question}\n\n### Assistant:"
        
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
                max_length=input_ids.shape[1] + 300,  # Generate up to 300 new tokens
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and clean up
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

        return generated_text.strip()

# Global instance
interview_prep_generator = InterviewPrepGenerator()

def analyze_answer_with_star(answer: str) -> str:
    """
    Analyzes a user's answer to an interview question using the STAR method with AI assistance.
    """
    if not interview_prep_generator.model_loaded:
        # Fallback to basic analysis
        feedback = "Thank you for your answer. "
        if "situation" in answer.lower() or "task" in answer.lower():
            feedback += "You've described the situation well. "
        if "action" in answer.lower():
            feedback += "You've clearly outlined the action you took. "
        if "result" in answer.lower():
            feedback += "It's great that you mentioned the result. "
        
        if len(feedback) < 50:
            return "That's a good start. Can you tell me more about the situation, the task you were assigned, the action you took, and the result of that action?"
        
        return feedback
    
    # Use AI model for analysis
    analysis_prompt = f"Analyze this interview answer using the STAR method (Situation, Task, Action, Result) and provide constructive feedback: {answer}"
    return interview_prep_generator.generate_response(analysis_prompt)

def generate_questions_from_cv(cv_data: Dict[str, Any], job_description: str) -> list[str]:
    """
    Generates interview questions based on the user's CV and the job description using AI.
    """
    if not interview_prep_generator.model_loaded:
        # Fallback to basic questions
        skills = ", ".join(cv_data.get('skills', []))
        return [
            f"Based on your experience at {cv_data.get('experiences', [{}])[0].get('company', 'your last company')}, can you describe a challenging project you worked on?",
            f"How have you used your skills in {skills} to solve a complex problem?",
            "What interests you most about this role?",
            "Describe a time you had to collaborate with a difficult team member.",
            "Where do you see yourself in five years?"
        ]
    
    # Use AI model to generate personalized questions
    skills = ", ".join(cv_data.get('skills', []))
    experiences = cv_data.get('experiences', [])
    cv_experience = "\n".join([f"- {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}" for exp in experiences]) if experiences else "Not available"
    
    question_prompt = f"Generate 5 interview questions based on this candidate profile:\nSkills: {skills}\nExperience: {cv_experience}\nJob Description: {job_description}"
    
    response = interview_prep_generator.generate_response(question_prompt)
    
    # Parse the response into individual questions
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if line and ('?' in line or line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•'))):
            # Clean up numbering and formatting
            question = line.replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '').replace('-', '').replace('•', '').strip()
            if question:
                questions.append(question)
    
    # Ensure we have at least some questions
    if len(questions) < 3:
        questions.extend([
            "Tell me about yourself.",
            "What are your strengths?",
            "Why do you want to work for this company?"
        ])
    
    return questions[:5]  # Return max 5 questions 