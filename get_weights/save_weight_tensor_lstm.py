import torch
import os
from pathlib import Path

# Path where checkpoints are stored
checkpoint_dir_str = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam"
checkpoint_files = sorted(os.listdir(checkpoint_dir_str))  # Ensure chronological order
checkpoint_dir = Path(checkpoint_dir_str)
weights_dir = checkpoint_dir / "weights" / "adam_training" # Define the weights directory

# Create the weights directory if it doesn't exist
weights_dir.mkdir(parents=True, exist_ok=True)

# Initialize lists to store tensors for each set of weights
embedding_weights = []
output_weights = []

# For each layer and gate type, create separate lists
# Layer 1 gates - Input to Hidden
layer1_forget_ih = []
layer1_input_ih = []
layer1_cell_ih = []
layer1_output_ih = []

# Layer 1 gates - Hidden to Hidden
layer1_forget_hh = []
layer1_input_hh = []
layer1_cell_hh = []
layer1_output_hh = []

# Layer 2 gates - Input to Hidden
layer2_forget_ih = []
layer2_input_ih = []
layer2_cell_ih = []
layer2_output_ih = []

# Layer 2 gates - Hidden to Hidden
layer2_forget_hh = []
layer2_input_hh = []
layer2_cell_hh = []
layer2_output_hh = []

# Loop through all items in the checkpoint directory
for item_name in checkpoint_files:
    item_path = checkpoint_dir / item_name

    # Check if the item is a file (and not the 'weights' directory)
    if item_path.is_file():
        try:
            checkpoint = torch.load(item_path, map_location="cpu")

            # Extract model weights from the checkpoint
            model_weights = checkpoint["model_state_dict"]

            # Extract and store embedding weights
            embedding_weights.append(model_weights["encoder.weight"])

            # Extract and store output weights
            output_weights.append(model_weights["decoder.weight"])

            # Process Layer 1
            weight_ih_l0 = model_weights["rnn.weight_ih_l0"]
            weight_hh_l0 = model_weights["rnn.weight_hh_l0"]

            # Split Layer 1 weights into gates (order defined here : https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
            W_i_ih_l0, W_f_ih_l0, W_c_ih_l0, W_o_ih_l0 = weight_ih_l0.chunk(4, dim=0)
            W_i_hh_l0, W_f_hh_l0, W_c_hh_l0, W_o_hh_l0 = weight_hh_l0.chunk(4, dim=0)

            # Append Layer 1 weights to respective lists
            layer1_forget_ih.append(W_f_ih_l0)
            layer1_input_ih.append(W_i_ih_l0)
            layer1_cell_ih.append(W_c_ih_l0)
            layer1_output_ih.append(W_o_ih_l0)

            layer1_forget_hh.append(W_f_hh_l0)
            layer1_input_hh.append(W_i_hh_l0)
            layer1_cell_hh.append(W_c_hh_l0)
            layer1_output_hh.append(W_o_hh_l0)

            # Process Layer 2
            weight_ih_l1 = model_weights["rnn.weight_ih_l1"]
            weight_hh_l1 = model_weights["rnn.weight_hh_l1"]

            # Split Layer 2 weights into gates
            W_i_ih_l1, W_f_ih_l1, W_c_ih_l1, W_o_ih_l1 = weight_ih_l1.chunk(4, dim=0)
            W_i_hh_l1, W_f_hh_l1, W_c_hh_l1, W_o_hh_l1 = weight_hh_l1.chunk(4, dim=0)

            # Append Layer 2 weights to respective lists
            layer2_forget_ih.append(W_f_ih_l1)
            layer2_input_ih.append(W_i_ih_l1)
            layer2_cell_ih.append(W_c_ih_l1)
            layer2_output_ih.append(W_o_ih_l1)

            layer2_forget_hh.append(W_f_hh_l1)
            layer2_input_hh.append(W_i_hh_l1)
            layer2_cell_hh.append(W_c_hh_l1)
            layer2_output_hh.append(W_o_hh_l1)

        except Exception as e:
            print(f"Error loading {item_name}: {e}")

# Save embedding and output weights
if embedding_weights:
    embedding_weights_tensor = torch.stack(embedding_weights)
    torch.save(embedding_weights_tensor, weights_dir / "embedding_weights.pt")

if output_weights:
    output_weights_tensor = torch.stack(output_weights)
    torch.save(output_weights_tensor, weights_dir / "output_weights.pt")

# Save Layer 1 gate weights
if layer1_forget_ih:
    torch.save(torch.stack(layer1_forget_ih), weights_dir / "layer1_forget_gate_ih.pt")
    torch.save(torch.stack(layer1_forget_hh), weights_dir / "layer1_forget_gate_hh.pt")
    torch.save(torch.stack(layer1_input_ih), weights_dir / "layer1_input_gate_ih.pt")
    torch.save(torch.stack(layer1_input_hh), weights_dir / "layer1_input_gate_hh.pt")
    torch.save(torch.stack(layer1_cell_ih), weights_dir / "layer1_cell_gate_ih.pt")
    torch.save(torch.stack(layer1_cell_hh), weights_dir / "layer1_cell_gate_hh.pt")
    torch.save(torch.stack(layer1_output_ih), weights_dir / "layer1_output_gate_ih.pt")
    torch.save(torch.stack(layer1_output_hh), weights_dir / "layer1_output_gate_hh.pt")

# Save Layer 2 gate weights
if layer2_forget_ih:
    torch.save(torch.stack(layer2_forget_ih), weights_dir / "layer2_forget_gate_ih.pt")
    torch.save(torch.stack(layer2_forget_hh), weights_dir / "layer2_forget_gate_hh.pt")
    torch.save(torch.stack(layer2_input_ih), weights_dir / "layer2_input_gate_ih.pt")
    torch.save(torch.stack(layer2_input_hh), weights_dir / "layer2_input_gate_hh.pt")
    torch.save(torch.stack(layer2_cell_ih), weights_dir / "layer2_cell_gate_ih.pt")
    torch.save(torch.stack(layer2_cell_hh), weights_dir / "layer2_cell_gate_hh.pt")
    torch.save(torch.stack(layer2_output_ih), weights_dir / "layer2_output_gate_ih.pt")
    torch.save(torch.stack(layer2_output_hh), weights_dir / "layer2_output_gate_hh.pt")

print("All valid checkpoint weights saved separately!")
