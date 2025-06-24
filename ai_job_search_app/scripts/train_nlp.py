import pandas as pd
import torch
import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from .preprocess_data import load_and_clean_data

class TrainingConfig:
    """Configuration for the training pipeline."""
    DATA_PATHS = ["../../Resume.csv", "../../UpdatedResumeDataSet.csv"]
    MODEL_NAME = 'bert-base-uncased'
    
    # Simplified local save paths
    MODEL_OUTPUT_PATH = '../../data/models/bert_resume_classifier'
    QUANTIZED_MODEL_PATH = '../../data/models/bert_resume_classifier_quantized'
    
    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCHS = 8
    LEARNING_RATE = 1e-5
    
    WEIGHT_DECAY = 0.02
    CLASSIFIER_DROPOUT = 0.4
    EARLY_STOPPING_PATIENCE = 2
    
    NUM_LAYERS_TO_FREEZE = 6
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        text, label = str(self.texts[item]), self.labels[item]
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label, dtype=torch.long)}

def train_epoch(model, data_loader, optimizer, device, scheduler, class_weights):
    model = model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(data_loader)

def eval_epoch(model, data_loader, device, class_weights):
    model = model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(data_loader), accuracy_score(all_labels, all_preds)

def quantize_and_save_model(model_path, quantized_model_path):
    if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
        print(f"ERROR: No saved model found at {model_path}. Cannot quantize.")
        return
        
    print(f"\nLoading best model from {model_path} for quantization...")
    model = BertForSequenceClassification.from_pretrained(model_path).to('cpu')
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("Quantization complete.")
    os.makedirs(quantized_model_path, exist_ok=True)
    torch.save(quantized_model.state_dict(), os.path.join(quantized_model_path, "pytorch_model.bin"))
    model.config.save_pretrained(quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")

def main():
    config = TrainingConfig()
    df = load_and_clean_data(config.DATA_PATHS)
    
    df['label'], label_names = pd.factorize(df['Category'])
    num_labels = len(label_names)
    id2label = {i: name for i, name in enumerate(label_names)}
    print(f"Found {num_labels} unique categories.")

    class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
    print("Calculated class weights to handle data imbalance.")
    
    X_train, X_val, y_train, y_val = train_test_split(df['cleaned_resume'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = ResumeDataset(X_train.values, y_train.values, tokenizer, config.MAX_LEN)
    val_dataset = ResumeDataset(X_val.values, y_val.values, tokenizer, config.MAX_LEN)
    
    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers)
    
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=num_labels,
        hidden_dropout_prob=config.CLASSIFIER_DROPOUT,
        attention_probs_dropout_prob=config.CLASSIFIER_DROPOUT,
        id2label=id2label
    ).to(config.DEVICE)
    
    if config.NUM_LAYERS_TO_FREEZE > 0:
        for param in model.bert.embeddings.parameters(): param.requires_grad = False
        for i in range(config.NUM_LAYERS_TO_FREEZE):
            for param in model.bert.encoder.layer[i].parameters(): param.requires_grad = False
        print(f"Froze embeddings and the first {config.NUM_LAYERS_TO_FREEZE} encoder layers.")
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, config.DEVICE, scheduler, class_weights)
        val_loss, val_accuracy = eval_epoch(model, val_loader, config.DEVICE, class_weights)
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"  **Validation loss improved. Saving model to {config.MODEL_OUTPUT_PATH}**")
            os.makedirs(config.MODEL_OUTPUT_PATH, exist_ok=True)
            model.save_pretrained(config.MODEL_OUTPUT_PATH)
            tokenizer.save_pretrained(config.MODEL_OUTPUT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered after {epoch+1} epochs.")
                break
    
    print(f"\n--- Training Complete ---")
    print(f"Best model (based on validation loss of {best_val_loss:.4f}) saved to {config.MODEL_OUTPUT_PATH}")
    
    quantize_and_save_model(config.MODEL_OUTPUT_PATH, config.QUANTIZED_MODEL_PATH)

if __name__ == '__main__':
    main() 