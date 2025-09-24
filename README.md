# Deep Learning Models for Sentiment Analysis on Disaster Tweets

This repository contains a comprehensive notebook demonstrating several deep learning methods to solve the **Disaster Tweets Challenge**. The notebook explores multiple models for sentiment analysis, including LSTM, BERT, XLNET, and RoBERTa, implemented using **PyTorch** and the Hugging Face Transformers library.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Presentation](#data-presentation)
3. [LSTM Classifier](#lstm-classifier)
4. [Fine-Tuning Transformer Models](#fine-tuning-transformer-models)
   - [BERT](#bert)
   - [XLNET](#xlnet)
   - [RoBERTa](#roberta)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results and Visualization](#results-and-visualization)
7. [Setup Instructions](#setup-instructions)
8. [Dependencies](#dependencies)

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
