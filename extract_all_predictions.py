import os
import subprocess

checkpoint_folder = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check"
input_folder = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp"
output_base = "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP"
start_checkpoint = "epoch_1_batch_12600"
start_processing = False
# Loop over all files in the checkpoint folder
for checkpoint in os.listdir(checkpoint_folder):
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint)

    if os.path.isfile(checkpoint_path) and checkpoint.endswith(".pt"):
        # Remove the .pt extension from the checkpoint name
        checkpoint_name = os.path.splitext(checkpoint)[0]

        if checkpoint_name == start_checkpoint:
            start_processing = True

        if start_processing:

            # Define the output directory based on the checkpoint name
            output_dir = os.path.join(output_base, checkpoint_name)

            # Call the extract_predictions.py script
            subprocess.run(
                [
                    "python",
                    "extract_predictions.py",
                    checkpoint_path,
                    "--i",
                    input_folder,
                    "--o",
                    output_dir,
                ]
            )
