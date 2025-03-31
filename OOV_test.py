def load_vocab(file_path):
    """Load a vocabulary file into a set."""
    with open(file_path, 'r', encoding='utf-8') as f:
        vocab = set(f.read().strip().split())  # Assuming one word per line
    return vocab

# Paths to the vocabulary files
train_vocab_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data/vocab.txt"
test_vocab_path = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/vocab.txt"

# Load the vocabularies
train_vocab = load_vocab(train_vocab_path)
test_vocab = load_vocab(test_vocab_path)

# Check if any test words are not in the training vocabulary
missing_words = test_vocab - train_vocab
if missing_words:
    print(f"Found {len(missing_words)} words in the test set that are not in the training set:")
    print(missing_words)
else:
    print("All words in the test set are present in the training set.")
