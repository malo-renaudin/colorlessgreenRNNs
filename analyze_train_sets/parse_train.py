import spacy
import benepar
from tqdm import tqdm
import json



nlp = spacy.load('en_core_web_sm')  # Smaller model
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# Disable unnecessary components for speed
nlp.disable_pipes(['ner', 'lemmatizer'])

file_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data/train.txt"

batch_size = 5000  # Much larger batches
line_count = 0

with open('parsed_results.txt', 'w') as out_f:  # Plain text, not JSON
    with open(file_path, 'r') as in_f:
        batch = []
        
        for line in tqdm(in_f):
            line = line.strip()
            if line:
                batch.append(line)
                
                if len(batch) == batch_size:
                    # Process entire batch at once
                    for doc in nlp.pipe(batch, batch_size=batch_size):
                        for sent in doc.sents:
                            # Write immediately, don't store in memory
                            out_f.write(f"{sent.text}\t{sent._.parse_string}\n")
                    
                    batch = []
                    line_count += batch_size
        
        # Process remaining
        if batch:
            for doc in nlp.pipe(batch):
                for sent in doc.sents:
                    out_f.write(f"{sent.text}\t{sent._.parse_string}\n")

print(f"Done. Output in parsed_results.txt")