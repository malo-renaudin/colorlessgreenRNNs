def create_vocab(input_file, vocab_file):
    vocab = set()

    # Read the nounpp_txt file and add words to the vocabulary set
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            words = line.strip().split()
            vocab.update(words)

    # Write the vocabulary to a file
    with open(vocab_file, 'w', encoding='utf8') as f:
        for word in sorted(vocab):  # Optionally sort the vocab
            f.write(f"{word}\n")

    print(f"Vocabulary saved to {vocab_file}")

# Usage
if __name__ == "__main__":
    # Specify your nounpp_txt file and desired vocab.txt output file
    nounpp_txt = '/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.text'  # Update with your actual file path
    vocab_file = 'nounpp_vocab.txt'  # Output vocab file

    # Create the vocabulary
    create_vocab(nounpp_txt, vocab_file)
