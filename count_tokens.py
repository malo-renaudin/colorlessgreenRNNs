import os

def count_tokens(path):
    """Counts the total number of tokens (words) in a text file."""
    assert os.path.exists(path)
    with open(path, 'r', encoding="utf8") as f:
        ntokens = 0
        for line in f:
            words = line.split()  # Tokenization at word level
            ntokens += len(words)
    return ntokens

# Example usage:
path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data/valid.txt"
total_tokens = count_tokens(path)
print(f"Total number of tokens in train.txt: {total_tokens}")