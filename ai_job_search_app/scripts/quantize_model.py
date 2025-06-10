import torch
import os
import json
from transformers import BertForSequenceClassification, BertConfig

def main():
    """
    Loads a fine-tuned model, applies dynamic quantization, and saves the optimized version.
    """
    class QuantizationConfig:
        MODEL_PATH = 'ai_job_search_app/data/models/bert_resume_classifier'
        QUANTIZED_MODEL_PATH = 'ai_job_search_app/data/models/bert_resume_classifier_quantized'
        DEVICE = torch.device("cpu")
        NUM_LABELS = 48 # As determined from our datasets

    config = QuantizationConfig()
    config_file_path = os.path.join(config.MODEL_PATH, 'config.json')

    # 1. Create config.json if it does not exist
    if not os.path.exists(config_file_path):
        print(f"Config file not found. Creating one at {config_file_path}")
        bert_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=config.NUM_LABELS)
        bert_config.save_pretrained(config.MODEL_PATH)

    # 2. Load the fine-tuned model
    print(f"Loading model from {config.MODEL_PATH}...")
    # The from_pretrained method will automatically find the .safetensors file
    model = BertForSequenceClassification.from_pretrained(config.MODEL_PATH).to(config.DEVICE)
    model.eval() # Set model to evaluation mode
    
    # 3. Apply Dynamic Quantization
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("Quantization complete.")

    # 4. Save the Quantized Model
    print(f"Saving quantized model to {config.QUANTIZED_MODEL_PATH}...")
    os.makedirs(config.QUANTIZED_MODEL_PATH, exist_ok=True)
    
    torch.save(quantized_model.state_dict(), os.path.join(config.QUANTIZED_MODEL_PATH, "pytorch_model.bin"))
    model.config.save_pretrained(config.QUANTIZED_MODEL_PATH)
    
    print("Quantized model saved successfully.")

if __name__ == '__main__':
    main() 