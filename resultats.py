
import os
import subprocess

# Define the checkpoint directory
checkpoints = '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/whole_model'

# Loop over all checkpoint files in the directory
for file in os.listdir(checkpoints):
    if file.endswith('.pt'):  # Assuming the checkpoint files have a .pt extension
        checkpoint_path = os.path.join(checkpoints, file)
        model_name = file.replace('.pt', '')  # Remove the .pt extension for the suffix

        # Construct the command to run
        cmd = [
            'python', '/scratch2/mrenaudin/colorlessgreenRNNs/src/language_models/evaluate_target_word.py',
            '--data', '/scratch2/mrenaudin/colorlessgreenRNNs/english_data/',
            '--checkpoint', checkpoint_path,
            '--path', '/scratch2/mrenaudin/colorlessgreenRNNs/data/agreement/English/generated',
            '--suffix', model_name
        ]

        # Print the command for debugging purposes
        print(f"Running command: {' '.join(cmd)}")

        # Run the command
        subprocess.run(cmd)

print("All checkpoints processed.")