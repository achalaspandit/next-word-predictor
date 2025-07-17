# Next Word Predictor

A next word prediction tool built with LSTM and Attention mechanism. This project trains a neural network to predict the next word in a sentence using Project Gutenberg Ebook "The Adventures of Sherlock Holmes" as the training corpus. It uses word level tokenization and embeddings from sentence transformers.


## Overview

This project includes:

- **`downloader.py`**: Downloads [EBOOK](https://www.gutenberg.org/files/1661/1661-0.txt)
- **`data_processor.py`**: Cleans, tokenizes, and prepares embeddings from the text.
- **`next-word-predictor.py`**: Trains a model and generates next-word predictions.
- **`demo.ipynb`**: Demo notebook with model trained outputs, implementing interactable version of next-word-predictor.py
---

## Architecture
The model consists of:

1. Embedding Layer: Converts tokenized words to dense vector embeddings generated with huggingface all-MiniLM-L6-v2 sentence embedding model 
2. LSTM Layer: 256 units and 2 layers for learning sequential patterns
3. Attention Layer: Computes weighted importance for each timestep in the sequence, focusing on the most relevant parts of the input.
Dense Output Layer: Predicts probability distribution over vocabulary

## How to Run

### 1. Clone Repository and move to project directory
```bash
git clone https://github.com/achalaspandit/next-word-predictor.git
cd next-word-predictor
```
### 2. Install Dependencies
Ensure [Poetry](https://python-poetry.org/docs/#installation) is installed, then run:

```bash
poetry install
```

### 3. Run predictor model
```bash
poetry run python next-word-predictor.py
```

What it does:

Downloads the text file (if not already available)
Processes the text into training data
Trains a prediction model
Outputs training loss update and next-word predictions for 5 different seed texts along with top 5 word predictions at each stage for the last seed text.

## Model Performance
Training Accuracy: 
Test Accuracy:
Perplexity:
Coherence:

Examples:


Key Differences across previous versions:
- Version 0: Byte Pair Encoding Tokenization and pytorch embedding layer.
  Train Accuracy: 15%
- Version 1: Word level encoding and custom embedding with all-MiniLM-L6-v2 sentence embedding model - reduced small vocabulary size.
  Train Accuracy: 93.6%, Test Accuracy: 77.9%, epochs: 100
  
- Version 2: Reduce repetitive words and temperature scaling 
  Train Accuracy: 95.92%, Test Accuracy: 80.31%, epochs: 125
  Coherence is much better when compared to Version 1

  <img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/ae91271e-c9f6-40ee-9510-0d49b7057065" />
  <img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/02f05e99-9c61-43f1-855e-3614dc6684a3" />

- Version 3: With regularization and top k sampling
  Train Accuracy: 96.55%, Test Accuracy: 81.33%, epochs: 130
  Coherence is comparable to version 2

  <img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/ac4f605d-8f36-4a22-bc1c-3400c2a0b7d3" />
  <img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/0cea0ed6-5945-4f2d-a1e7-1d0539b04c29" />

- Version 4: Single LSTM layer and top k sampling


