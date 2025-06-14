import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

class MLService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        # --- Paths ---
        # Assuming this file is in ai_job_search_app/backend/services/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        quantized_model_path = os.path.join(project_root, 'data', 'models', 'bert_resume_classifier_quantized')
        tokenizer_path = os.path.join(project_root, 'data', 'models', 'bert_resume_classifier')

        # --- Load Model and Tokenizer ---
        if not os.path.exists(quantized_model_path):
            raise RuntimeError(f"Quantized model not found at {quantized_model_path}. Please run the quantization script.")
        
        if not os.path.exists(tokenizer_path):
             raise RuntimeError(f"Tokenizer files not found at {tokenizer_path}.")
        
        print("Loading ML model and tokenizer...")
        
        # Load the configuration from the quantized path, and the state dict
        self.model = BertForSequenceClassification.from_pretrained(quantized_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True
        print("ML Service Initialized.")

    def predict(self, text: str) -> str:
        """
        Predicts the category of a given text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        # The model's config contains the label mappings
        predicted_label = self.model.config.id2label[predicted_class_id]
        
        return predicted_label

# Singleton instance
ml_service = MLService() 