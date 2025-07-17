# Next Word Predictor

A next-word prediction tool built with LSTM and Attention layer. It uses word level tokenization and embeddings from sentence transformers. The model is trained with The Project Gutenberg Ebook The Adventures of Sherlock Holmes.


## Overview

This project includes:

- **`downloader.py`**: Downloads [EBOOK](https://www.gutenberg.org/files/1661/1661-0.txt)
- **`data_processor.py`**: Cleans, tokenizes, and prepares embeddings from the text.
- **`next-word-predictor.py`**: Trains a model and generates next-word predictions.
- **`demo.ipynb`**: Demo notebook with model trained outputs.
---

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
