# PersianPoetryGenerator

A complete deep learning pipeline for generating Persian poetry using LSTM, Transformer, and fine-tuned GPT-2 models. The project includes text preprocessing, model training (character-level and word-level), and generation with control over creativity (via temperature, top-k, top-p, and repetition penalty).

---

## Objective

To build an advanced Persian poetry generator trained on classical texts using various neural network architectures.

---

##  Directory Overview

```
TextGenerator/
├── Dataset/               # Raw & cleaned Persian poems
│   ├── persianPoet.txt
│   ├── naserkhosro.txt
│   ├── attar.txt
│   └── merged_poets.txt
├── FastText/             # FastText embeddings (cc.fa.300.vec)
├── Model/                # Model visualizations or exports
├── TrainingCheckpoints/ # Trained models and weight checkpoints
├── TextGenerator.ipynb   # All preprocessing + training + generation
├── README.md             # This file
```

---

## Workflow Breakdown

### 1. Text Preprocessing
- Normalization of punctuation and digits
- Cleaning invalid characters
- Merging multiple poetry sources into a unified dataset

### 2. Character-Level LSTM
- Input: 100-character sequences
- Basic LSTM and Advanced LSTM (BatchNorm, Dropout, ElasticNet)
- Checkpointed training with early stopping

### 3. Transformer Model
- From scratch using Positional Encoding and Multi-Head Attention
- FastText (Persian) embeddings used
- Trained at both char- and word-level

### 4. GPT-2 Fine-tuning (HuggingFace)
- Pretrained model: `HooshvareLab/gpt2-fa`
- Tokenized with `transformers` tokenizer
- Trainer API with early stopping and checkpointing
- Final model saved and used for text generation

---

## How to Run

> All scripts and training were done inside `TextGenerator.ipynb`. You can split into separate `.py` files if needed.

### Clone and Prepare
```bash
git clone https://github.com/masoumehkhaleghian/TextGenerator.git
cd TextGenerator
```

### Launch Notebook
```bash
jupyter notebook TextGenerator.ipynb
```

---

## Model Checkpoints
All trained model files and intermediate checkpoints (LSTM, Transformer, GPT-2) are saved under:
```
TrainingCheckpoints/
```
For example:
- `best_model.weights.h5`
- `best_model_transformer.keras`
- `poetry-gpt2-fa/checkpoint-XXXXX`

---

## Requirements
```txt
tensorflow>=2.11
numpy
pandas
re
transformers
datasets
accelerate
matplotlib
```

---

## Sample Output
```text
ای یار، دریغا که لاحول عشق
تو را در دل ِ جان و دل نيست زآن
```
---

## Resources
-  Full project and data: [Google Drive](https://drive.google.com/drive/folders/1dvj6VtVJiGaJWPp1R2T5f66Yi8OQjhjw?usp=sharing)
-  GPT-2 model: https://huggingface.co/HooshvareLab/gpt2-fa

---

## Author
**Masoumeh Khaleghian**  
Persian NLP Engineer and AI Researcher  
GitHub: [masoumehkhaleghian](https://github.com/masoumehkhaleghian)

---

> If you use this repo or model, consider citing or mentioning it in your work!
