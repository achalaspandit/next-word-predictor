import re
import random
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import pickle
import os

def create_word_embeddings_and_tokenization(input_file, train_ratio=0.7, val_ratio=0.15):
    """
    Tokenize text and create word embeddings using SentenceTransformers
    
    Args:
        input_file: Path to the text file
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
    
    Returns:
        train_sequences, val_sequences, test_sequences, word_to_idx, idx_to_word, embeddings_matrix
    """

    # Read and clean the content
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the book content between the standard Project Gutenberg markers
    splits = content.split("*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK\nHOLMES ***")
    splits = splits[1].split("*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK\nHOLMES ***")
    book_content = splits[0].replace("\r\n", " ").replace("\n", " ").replace("£", "pounds").replace("½", "one half").replace("&", "and").replace(":", "").replace("(", "").replace(")", "").replace("—", " ").replace("–", " ").replace(";", "").replace("_", " ").replace("à", "a").replace( 'â', "a").replace('æ', "ae").replace('è', "e").replace('é', "e").replace('œ', "oe").replace('‘', "").replace('’', "").replace('“', "").replace('”', "").strip()

    # Clean up whitespace and split sentences
    book_content = re.sub(r'\s+', ' ', book_content).strip()
    sentences = re.split(r'(?<=[.!?])\s+', book_content)

    # Create chunks
    i = 0
    chunks = []
    c = ""
    while i < len(sentences):
        c = c + " " + sentences[i]
        if len(c) > 150:
            chunks.append(c)
            c = sentences[i]
        i += 1

    # Create train val test splits 
    random.shuffle(chunks)
    total_chunks = len(chunks)
    train_split = int(total_chunks * train_ratio)
    val_split = int(total_chunks * (train_ratio + val_ratio))
    train_chunks = chunks[:train_split]
    val_chunks = chunks[train_split:val_split]
    test_chunks = chunks[val_split:]

    # WORD-LEVEL TOKENIZATION
    def custom_tokenize_with_spaces(chunks):
        chunked_tokens = []
        for sent in chunks:
          sent = sent.lower()
          tokens = re.findall(r'\w+|[^\w\s]|\s+', sent)
          chunked_tokens.append(tokens)
        return chunked_tokens

    train_tokenized = custom_tokenize_with_spaces(train_chunks)
    val_tokenized = custom_tokenize_with_spaces(val_chunks)
    test_tokenized = custom_tokenize_with_spaces(test_chunks)

    #Build Vocabulary
    word_counts = Counter()
    for chunk in train_tokenized:
        word_counts.update(chunk)
    
    for chunk in val_tokenized:
        word_counts.update(chunk)

    for chunk in test_tokenized:
        word_counts.update(chunk)

    vocab_words = [word for word, count in word_counts.items() if count > 2 ]
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    vocab_words = special_tokens + vocab_words

    word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # EMBEDDING GENERATION FOR VOCAB
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings_list = []
    embedding_dim = model.encode(['the']).shape[1]

    for word in vocab_words:
        if word in special_tokens:
            if word == '<PAD>':
                embedding = np.zeros(embedding_dim)
            elif word == '<UNK>': # Random vector for unknown words
                embedding = np.random.normal(0, 0.1, embedding_dim)
            elif word == '<START>': # Random vector for start token
                embedding = np.random.normal(0, 0.1, embedding_dim)
            elif word == '<END>': # Random vector for end token
                embedding = np.random.normal(0, 0.1, embedding_dim)
        else:
            embedding = model.encode([word])[0]
        embeddings_list.append(embedding)
    embeddings_matrix = np.vstack(embeddings_list)

    # CONVERT TOKENIZED CHUNKS TO SEQUENCES
    def words_to_sequences(tokenized_chunks, word_to_idx):
        sequences = []
        for chunk in tokenized_chunks:
            sequence = []
            for word in chunk:
                if word in word_to_idx:
                    sequence.append(word_to_idx[word])
                else:
                    sequence.append(word_to_idx['<UNK>'])
            sequences.append(sequence)
        return sequences

    train_sequences = words_to_sequences(train_tokenized, word_to_idx)
    val_sequences = words_to_sequences(val_tokenized, word_to_idx)
    test_sequences = words_to_sequences(test_tokenized, word_to_idx)

    return train_sequences, val_sequences, test_sequences, word_to_idx, idx_to_word, embeddings_matrix

    


def prepare_data_and_embeddings(input_file="1661-0.txt", force_recreate=False):
    """
    Main function to prepare data and embeddings if don't exist.
    
    Args:
        input_file (str): Path to input text file
        force_recreate (bool): Force recreation even if files exist
        
    Returns:
        bool: True if processing was done, False if files already existed
    """
    required_files = ["embeddings.pkl", "data.pkl"]
    
    if not force_recreate and all(os.path.exists(file_path) for file_path in required_files):
        print("Data files and tokenizer already exist. Skipping processing.")
        return False
    
    print("Processing text file and creating data splits...")

    train_sequences, val_sequences, test_sequences, word_to_idx, idx_to_word, embeddings_matrix = create_word_embeddings_and_tokenization(input_file)
    
    data = {
    'train_sequences': train_sequences,
    'val_sequences': val_sequences,
    'test_sequences': test_sequences
    }
    embeddings = {
    'word_to_idx': word_to_idx,
    'idx_to_word': idx_to_word,
    'embeddings_matrix': embeddings_matrix
    }

    with open("embeddings.pkl", 'wb') as f:
        pickle.dump(embeddings, f)

    with open("data.pkl", 'wb') as f:
        pickle.dump(data, f)
        
    print("Data processing complete. Files saved as 'embeddings.pkl' and 'data.pkl'.")
    return True


if __name__ == "__main__":
    prepare_data_and_embeddings()