import torch
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

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
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")
        self.fallback_mode = False
        
        # Try to load the BERT model with better error handling
        try:
            self._load_bert_model()
            logger.info("ML Service initialized with BERT model")
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            logger.info("ML Service initialized in fallback mode")
            self.fallback_mode = True
        
        self.initialized = True

    def _load_bert_model(self):
        """Load BERT model with proper error handling"""
        try:
            # Import here to avoid global import issues
            from transformers import BertTokenizer, BertForSequenceClassification
            
            # --- Paths ---
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            quantized_model_path = os.path.join(project_root, 'data', 'models', 'bert_resume_classifier_quantized')
            tokenizer_path = os.path.join(project_root, 'data', 'models', 'bert_resume_classifier')

            # Check if model files exist
            if not os.path.exists(quantized_model_path):
                raise FileNotFoundError(f"Quantized model not found at {quantized_model_path}")
            
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer files not found at {tokenizer_path}")
            
            # Load model and tokenizer
            self.model = BertForSequenceClassification.from_pretrained(quantized_model_path)
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            self.model.to(self.device)
            self.model.eval()
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise

    def predict(self, text: str) -> str:
        """
        Predicts the category of a given text.
        Falls back to rule-based classification if BERT model is unavailable.
        """
        if not self.fallback_mode and self.model and self.tokenizer:
            try:
                return self._predict_with_bert(text)
            except Exception as e:
                logger.warning(f"BERT prediction failed: {e}, falling back to rule-based")
                return self._predict_with_rules(text)
        else:
            return self._predict_with_rules(text)

    def _predict_with_bert(self, text: str) -> str:
        """Predict using BERT model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        # The model's config contains the label mappings
        if hasattr(self.model.config, 'id2label'):
            predicted_label = self.model.config.id2label[predicted_class_id]
        else:
            # Fallback labels if config doesn't have them
            labels = ['HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE', 
                     'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 
                     'BPO', 'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 
                     'CHEF', 'FINANCE', 'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION']
            predicted_label = labels[predicted_class_id] if predicted_class_id < len(labels) else 'GENERAL'
        
        return predicted_label

    def _predict_with_rules(self, text: str) -> str:
        """Rule-based classification fallback"""
        text_lower = text.lower()
        
        # Define keyword patterns for different categories
        category_keywords = {
            'INFORMATION-TECHNOLOGY': [
                'python', 'java', 'javascript', 'programming', 'software', 'developer', 
                'engineer', 'coding', 'algorithm', 'database', 'api', 'frontend', 
                'backend', 'fullstack', 'devops', 'cloud', 'aws', 'azure', 'docker'
            ],
            'DESIGNER': [
                'design', 'ui', 'ux', 'graphic', 'creative', 'photoshop', 'illustrator', 
                'figma', 'sketch', 'branding', 'visual', 'layout', 'typography'
            ],
            'FINANCE': [
                'finance', 'accounting', 'financial', 'budget', 'investment', 'banking', 
                'audit', 'tax', 'revenue', 'profit', 'excel', 'financial analysis'
            ],
            'HEALTHCARE': [
                'medical', 'healthcare', 'nurse', 'doctor', 'hospital', 'patient', 
                'clinical', 'pharmacy', 'treatment', 'diagnosis', 'therapy'
            ],
            'SALES': [
                'sales', 'marketing', 'customer', 'client', 'revenue', 'target', 
                'lead', 'prospect', 'negotiation', 'relationship', 'crm'
            ],
            'BUSINESS-DEVELOPMENT': [
                'business', 'strategy', 'development', 'growth', 'partnership', 
                'market', 'analysis', 'planning', 'management', 'operations'
            ],
            'HR': [
                'human resources', 'hr', 'recruitment', 'hiring', 'talent', 
                'employee', 'payroll', 'benefits', 'training', 'performance'
            ],
            'TEACHER': [
                'teacher', 'education', 'teaching', 'instructor', 'professor', 
                'curriculum', 'student', 'learning', 'academic', 'school'
            ],
            'ENGINEERING': [
                'engineering', 'engineer', 'mechanical', 'electrical', 'civil', 
                'construction', 'manufacturing', 'technical', 'project', 'design'
            ]
        }
        
        # Score each category based on keyword matches
        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score
        
        # Return the category with the highest score
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'GENERAL'

    def get_model_info(self) -> dict:
        """Get information about the current model state"""
        return {
            'initialized': self.initialized,
            'fallback_mode': self.fallback_mode,
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'device': str(self.device)
        }

# Singleton instance
ml_service = MLService() 