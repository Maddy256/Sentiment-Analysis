# Deep Learning Models for Sentiment Analysis on Disaster Tweets

This repository contains a comprehensive notebook demonstrating several deep learning methods to solve the **Disaster Tweets Challenge**. The notebook explores multiple models for sentiment analysis, including LSTM, BERT, XLNET, and RoBERTa, implemented using **PyTorch** and the Hugging Face Transformers library.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Presentation](#data-presentation)
3. [LSTM Classifier](#LSTM-classifier)
4. [BERT Fine-tuning](#BERT-Fine-tuning)
5. [Ensemble Prediction](#Ensemble-Prediction)
6. [Results ](#results)

---

## Project Overview

The goal is to classify tweets into **disaster-related** or **non-disaster-related** using deep learning models. The notebook demonstrates:

- Data preprocessing
- LSTM-based sentiment classification
- Fine-tuning of transformer models (BERT, XLNET, RoBERTa)
- Model evaluation and comparison
- Prediction on test datasets

---

## Data Presentation

The dataset contains the following columns:

| Column    | Description |
|-----------|-------------|
| id        | Tweet ID    |
| keyword   | Relevant keyword (if any) |
| location  | Location of the tweet |
| text      | Text content of the tweet |
| target    | Sentiment label: 1 for disaster, 0 for non-disaster |

```python
import pandas as pd
df = pd.read_csv(path_data_train)
df.head()
```
## LSTM Classifier
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=True, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                  batch_first=True, enforce_sorted=False)
        output, (hidden, _) = self.lstm(packed)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(hidden))
```
# Training LSTM
```python
model = LSTMClassifier(vocab_size=10000, embedding_dim=100, hidden_dim=64, output_dim=1, n_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Train for 12 epochs
for epoch in range(12):
    # Training loop here
    pass
```
## BERT Fine-tuning
```python
def train_transformer_model(model_name, epochs=2):
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=50)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["id", "keyword", "location", "text"])
    tokenized_datasets.set_format("torch")
    
    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # Data loaders
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=32)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=32)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader):.4f}")
    
    return model
```
## Ensemble Prediction
```python
def ensemble_predictions(models, tokenizer, test_data, weights):
    """Combine predictions from multiple models using weighted averaging"""
    
    all_probabilities = []
    
    for model in models:
        model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch in test_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]  # Positive class probability
                probabilities.extend(probs.cpu().numpy())
        
        all_probabilities.append(probabilities)
    
    # Weighted ensemble
    final_probs = np.average(all_probabilities, axis=0, weights=weights)
    predictions = (final_probs > 0.5).astype(int)
    
    return predictions, final_probs
    ```

# Create ensemble
```python
models = [bert_model, roberta_model, xlnet_model]
weights = [0.33, 0.34, 0.33]  # Based on validation F1 scores
ensemble_preds, ensemble_probs = ensemble_predictions(models, tokenizer, test_datal
```



## 📊 Results

| Model   | Accuracy / F1 | Notes                          |
|---------|---------------|--------------------------------|
| LSTM    | ~78%          | Basic model with word embeddings |
| BERT    | ~83%          | HuggingFace pretrained         |
| RoBERTa | ~83%          | Robust results                 |
| XLNet   | ~81%          | Context-aware but heavier      |
| Bagging | 84%           | Best performance               |
