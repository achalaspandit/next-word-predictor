# Next Word Predictor

A next word prediction tool built with LSTM and Attention mechanism. This project trains a neural network to predict the next word in a sentence using Project Gutenberg Ebook "The Adventures of Sherlock Holmes" as the training corpus. It uses word level tokenization and embeddings from sentence transformers.


## Overview

This project includes:
- **`downloader.py`**: Downloads [EBOOK](https://www.gutenberg.org/files/1661/1661-0.txt)
- **`data_processor.py`**: Cleans, tokenizes, and prepares embeddings from the text.
- **`next-word-predictor.py`**: Trains a model and generates next-word predictions.
- **`demo.ipynb`**: Demo notebook with model trained outputs, implementing interactable version of next-word-predictor.py


## Architecture
The model consists of:
1. Embedding Layer: Converts tokenized words to dense vector embeddings generated with huggingface all-MiniLM-L6-v2 sentence embedding model 
2. LSTM Layer: 256 units and 2 layers for learning sequential patterns
3. Attention Layer: Computes weighted importance for each timestep in the sequence, focusing on the most relevant parts of the input.
4. Dense Output Layer: Predicts probability distribution over vocabulary
<img width="871" height="101" alt="arch" src="https://github.com/user-attachments/assets/afb70fa7-7917-44bd-8ade-0e6e25ccce72" />

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

#### What it does:

1. Downloads the text file (if not already available)
2. Processes the text into training data
3. Trains a prediction model
4. Outputs training loss update and next-word predictions for 5 different seed texts along with top 5 word predictions at each stage for the last seed text.

## Model Performance
#### Training Accuracy: 95.92%

#### Test Accuracy: 80.31%

#### Perplexity: 3.34

#### Examples:
Example 1:

Generating from: 'Boots which extended halfway'

Generated text: boots which extended halfway over all that press and held a pounds work. what took place my earth, said he and! a secret was in his

Example 2:

Generating from: 'I ordered him to pay'

Generated text: i ordered him to pay over fifty pounds 1000 work. he was on professional business in discovering half, for a week after with his dreams and land when

Example 3:

Generating from: 'I answered that it had'

Generated text: i answered that it had been conviction if she carried out from his death but taken up by history and never mind to get for his work in my

Example 4: 

Generating from: 'remove crusted mud from it. Hence'

Generated text: remove mud from it. hence ? well, how took to doctors round it, and so! what do my business knew how give me to buy his mother!

Example 5:

Generating from: 'He never spoke of the'

Generated text: he never spoke of the manner before him of life

--- Prediction Details ---
Step 1:
Top 5 likely words:   (1.0000), - (0.0000), silence (0.0000), unpleasant (0.0000), ! (0.0000)

Step 2:
Top 5 likely words: bridegroom (0.3350), manner (0.2153), words (0.1213), death (0.0664), silence (0.0493)

Step 3:
Top 5 likely words:   (0.9917), , (0.0055), . (0.0016), death (0.0009), ! (0.0002)

Step 4:
Top 5 likely words: himself (0.6891), never (0.2624), of (0.0239), which (0.0107), before (0.0034)

Step 5:
Top 5 likely words:   (0.9976), , (0.0023), ? (0.0000), ! (0.0000), . (0.0000)

Step 6:'
Top 5 likely words: never (0.4794), he (0.2339), himself (0.1464), it (0.0485), him (0.0427)

Step 7:
Top 5 likely words:   (0.9796), . (0.0174), , (0.0019), ? (0.0012), ! (0.0000)

Step 8:
Top 5 likely words: of (0.6853), for (0.0783), also (0.0774), in (0.0452), never (0.0243)

Step 9:
Top 5 likely words:   (1.0000), - (0.0000), ? (0.0000), , (0.0000), . (0.0000)

Step 10:
Top 5 likely words: life (0.9782), strange (0.0063), himself (0.0051), <UNK> (0.0025), course (0.0015)



#### Key Differences across previous versions:
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
  Train Accuracy:94.74% , Test Accuracy:80.40% , epochs:100

I reduced LSTM layers from 2 to 1 that made the overall model less complex helping in faster convergence while not compromising on accuracy and coherence.
