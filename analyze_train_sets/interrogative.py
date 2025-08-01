def count_interrogative_sentences(file_path):
    """
    Count the number and percentage of interrogative sentences in a corpus.
    """
    total_sentences = 0
    interrogative_count = 0
    
    # All wh-words (upper and lower case)
    # wh_words = ['who', 'what', 'where', 'when', 'why', 'which', 'whose', 'whom', 'how',
    #             'whatever', 'whoever', 'wherever', 'whenever', 'whichever', 'whomever', 'however',
    #             'Who', 'What', 'Where', 'When', 'Why', 'Which', 'Whose', 'Whom', 'How',
    #             'Whatever', 'Whoever', 'Wherever', 'Whenever', 'Whichever', 'Whomever', 'However']
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().replace('<eos>', '').strip()
                if line:
                    total_sentences += 1
                    if line.endswith('?'):
                        interrogative_count += 1
        
        percentage = (interrogative_count / total_sentences * 100) if total_sentences > 0 else 0
        print(f"Total sentences: {total_sentences:,}")
        print(f"Interrogative sentences: {interrogative_count:,}")
        print(f"Percentage of interrogative sentences: {percentage:.2f}%")
        return total_sentences, interrogative_count, percentage
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 0, 0, 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0, 0, 0

# Usage
if __name__ == "__main__":
    file_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data/train.txt"
    count_interrogative_sentences(file_path)