import torch
import os
from pathlib import Path

# Path where checkpoints are stored
checkpoint_dir_str = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check_shuffled"
checkpoint_files =  [f'epoch_1_batch_{i}.pt' for i in range(0, 301, 1)] + [f'epoch_1_batch_{i}.pt'for i in range(400, 9300, 100)]+[f'epoch_{i}.pt' for i in range(1, 41, 1)]
#[f'epoch_1_batch_{i}.pt' for i in range(0, 101, 1)] 
#[f'epoch_1_batch_{i}.pt'for i in range(200, 9300, 100)]
checkpoint_dir = Path(checkpoint_dir_str)
weights_dir = checkpoint_dir / "weights" # Define the weights directory

# Create the weights directory if it doesn't exist
weights_dir.mkdir(parents=True, exist_ok=True)

# # Initialize lists to store tensors for each set of weights
# embedding_weights = []
# output_weights = []

# # For each layer and gate type, create separate lists
# # Layer 1 gates - Input to Hidden
# layer1_forget_ih = []
# layer1_input_ih = []
# layer1_cell_ih = []
# layer1_output_ih = []

# # Layer 1 gates - Hidden to Hidden
# layer1_forget_hh = []
# layer1_input_hh = []
# layer1_cell_hh = []
# layer1_output_hh = []

# # Layer 2 gates - Input to Hidden
# layer2_forget_ih = []
# layer2_input_ih = []
# layer2_cell_ih = []
# layer2_output_ih = []

# # Layer 2 gates - Hidden to Hidden
# layer2_forget_hh = []
# layer2_input_hh = []
# layer2_cell_hh = []
# layer2_output_hh = []

# Loop through all items in the checkpoint directory
for item_name in checkpoint_files:
    item_path = checkpoint_dir / item_name

    # Check if the item is a file (and not the 'weights' directory)
    if item_path.is_file():
        try:
            print(item_path)
            checkpoint = torch.load(item_path, map_location="cpu")

            # Extract model weights from the checkpoint
            model_weights = checkpoint["model_state_dict"]

            # Extract and store embedding weights
            embedding = model_weights["encoder.weight"]
            embedding_file = weights_dir / f"embedding_weights_{item_name}.pt"
            torch.save(embedding, embedding_file)

            # Extract and store output weights
            output = model_weights["decoder.weight"]
            output_file = weights_dir / f"output_weights_{item_name}.pt"
            torch.save(output, output_file)

            # Process Layer 1
            weight_ih_l0 = model_weights["rnn.weight_ih_l0"]
            weight_hh_l0 = model_weights["rnn.weight_hh_l0"]
            W_i_ih_l0, W_f_ih_l0, W_c_ih_l0, W_o_ih_l0 = weight_ih_l0.chunk(4, dim=0)
            W_i_hh_l0, W_f_hh_l0, W_c_hh_l0, W_o_hh_l0 = weight_hh_l0.chunk(4, dim=0)

            torch.save(W_f_ih_l0, weights_dir / f"layer1_forget_gate_ih_{item_name}.pt")
            torch.save(W_i_ih_l0, weights_dir / f"layer1_input_gate_ih_{item_name}.pt")
            torch.save(W_c_ih_l0, weights_dir / f"layer1_cell_gate_ih_{item_name}.pt")
            torch.save(W_o_ih_l0, weights_dir / f"layer1_output_gate_ih_{item_name}.pt")

            torch.save(W_f_hh_l0, weights_dir / f"layer1_forget_gate_hh_{item_name}.pt")
            torch.save(W_i_hh_l0, weights_dir / f"layer1_input_gate_hh_{item_name}.pt")
            torch.save(W_c_hh_l0, weights_dir / f"layer1_cell_gate_hh_{item_name}.pt")
            torch.save(W_o_hh_l0, weights_dir / f"layer1_output_gate_hh_{item_name}.pt")

            # Process Layer 2
            weight_ih_l1 = model_weights["rnn.weight_ih_l1"]
            weight_hh_l1 = model_weights["rnn.weight_hh_l1"]
            W_i_ih_l1, W_f_ih_l1, W_c_ih_l1, W_o_ih_l1 = weight_ih_l1.chunk(4, dim=0)
            W_i_hh_l1, W_f_hh_l1, W_c_hh_l1, W_o_hh_l1 = weight_hh_l1.chunk(4, dim=0)

            torch.save(W_f_ih_l1, weights_dir / f"layer2_forget_gate_ih_{item_name}.pt")
            torch.save(W_i_ih_l1, weights_dir / f"layer2_input_gate_ih_{item_name}.pt")
            torch.save(W_c_ih_l1, weights_dir / f"layer2_cell_gate_ih_{item_name}.pt")
            torch.save(W_o_ih_l1, weights_dir / f"layer2_output_gate_ih_{item_name}.pt")

            torch.save(W_f_hh_l1, weights_dir / f"layer2_forget_gate_hh_{item_name}.pt")
            torch.save(W_i_hh_l1, weights_dir / f"layer2_input_gate_hh_{item_name}.pt")
            torch.save(W_c_hh_l1, weights_dir / f"layer2_cell_gate_hh_{item_name}.pt")
            torch.save(W_o_hh_l1, weights_dir / f"layer2_output_gate_hh_{item_name}.pt")

        except Exception as e:
            print(f"Error loading {item_name}: {e}")

# # Save embedding and output weights
# if embedding_weights:
#     embedding_weights_tensor = torch.stack(embedding_weights)
#     torch.save(embedding_weights_tensor, weights_dir / "embedding_weights.pt")

# if output_weights:
#     output_weights_tensor = torch.stack(output_weights)
#     torch.save(output_weights_tensor, weights_dir / "output_weights.pt")

# #Save Layer 1 gate weights
# if layer1_forget_ih:
#     torch.save(torch.stack(layer1_forget_ih), weights_dir / "layer1_forget_gate_ih_ep.pt")
#     torch.save(torch.stack(layer1_forget_hh), weights_dir / "layer1_forget_gate_hh_ep.pt")
#     torch.save(torch.stack(layer1_input_ih), weights_dir / "layer1_input_gate_ih_ep.pt")
#     torch.save(torch.stack(layer1_input_hh), weights_dir / "layer1_input_gate_hh_ep.pt")
#     torch.save(torch.stack(layer1_cell_ih), weights_dir / "layer1_cell_gate_ih_ep.pt")
#     torch.save(torch.stack(layer1_cell_hh), weights_dir / "layer1_cell_gate_hh_ep.pt")
#     torch.save(torch.stack(layer1_output_ih), weights_dir / "layer1_output_gate_ih_ep.pt")
#     torch.save(torch.stack(layer1_output_hh), weights_dir / "layer1_output_gate_hh_ep.pt")

# # Save Layer 2 gate weights
# if layer2_forget_ih:
#     torch.save(torch.stack(layer2_forget_ih), weights_dir / "layer2_forget_gate_ih_ep.pt")
#     torch.save(torch.stack(layer2_forget_hh), weights_dir / "layer2_forget_gate_hh_ep.pt")
#     torch.save(torch.stack(layer2_input_ih), weights_dir / "layer2_input_gate_ih_ep.pt")
#     torch.save(torch.stack(layer2_input_hh), weights_dir / "layer2_input_gate_hh_ep.pt")
#     torch.save(torch.stack(layer2_cell_ih), weights_dir / "layer2_cell_gate_ih_ep.pt")
#     torch.save(torch.stack(layer2_cell_hh), weights_dir / "layer2_cell_gate_hh_ep.pt")
#     torch.save(torch.stack(layer2_output_ih), weights_dir / "layer2_output_gate_ih_ep.pt")
#     torch.save(torch.stack(layer2_output_hh), weights_dir / "layer2_output_gate_hh_ep.pt")

print("All valid checkpoint weights saved separately!")
