import torch
import os

# Path where checkpoints are stored
checkpoint_dir = "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/full_check"
checkpoint_files = sorted(os.listdir(checkpoint_dir))  # Ensure chronological order

# Initialize lists to store tensors for each set of weights
embedding_weights = []
output_weights = []
layer_1_gate_weights = []
layer_2_gate_weights = []

# Loop through all checkpoint files
for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model weights from the checkpoint
    model_weights = checkpoint

    # Extract and store embedding weights
    embedding_weights.append(model_weights['encoder.weight'])

    # Extract and store output weights
    output_weights.append(model_weights['decoder.weight'])
    #output_bias.append(model_weights['decoder.bias'])

    # Initialize lists to store gate weights for each layer
    layer_1_weights = []
    layer_2_weights = []

    # Loop through each LSTM layer (assuming 2 layers)
    for layer in range(2):
        # Extract the input-to-hidden (weight_ih) and hidden-to-hidden (weight_hh) weights for the layer
        weight_ih = model_weights[f'rnn.weight_ih_l{layer}']  # Input-to-hidden
        weight_hh = model_weights[f'rnn.weight_hh_l{layer}']  # Hidden-to-hidden

        # Compute hidden_size by dividing the number of rows in weight_ih by 4 (since there are 4 gates)
        hidden_size = weight_ih.shape[0] // 4

        # Split the weight matrices into individual gates (forget, input, cell, output gates)
        W_f_ih, W_i_ih, W_c_ih, W_o_ih = weight_ih.chunk(4, dim=0)
        W_f_hh, W_i_hh, W_c_hh, W_o_hh = weight_hh.chunk(4, dim=0)

        # Concatenate input-to-hidden and hidden-to-hidden weights for each gate and add to list
        layer_gate_weights = [
            torch.cat((W_f_ih, W_f_hh), dim=1),  # Forget gate
            torch.cat((W_i_ih, W_i_hh), dim=1),  # Input gate
            torch.cat((W_c_ih, W_c_hh), dim=1),  # Cell state gate
            torch.cat((W_o_ih, W_o_hh), dim=1)   # Output gate
        ]
        
        # Append the gate weights for this layer
        if layer == 0:
            layer_1_weights.append(torch.stack(layer_gate_weights))
        else:
            layer_2_weights.append(torch.stack(layer_gate_weights))

    # Stack layer weights for both layers
    layer_1_gate_weights.append(torch.stack(layer_1_weights))  # Stack gates for layer 1
    layer_2_gate_weights.append(torch.stack(layer_2_weights))  # Stack gates for layer 2
# Convert lists to tensors
embedding_weights_tensor = torch.stack(embedding_weights)# Shape: (num_checkpoints, embedding_size)
output_weights_tensor = torch.stack(output_weights)  # Shape: (num_checkpoints, output_size)
layer_1_weights_tensor = torch.stack(layer_1_gate_weights)  # Shape: (num_checkpoints, 4 gates, hidden_size, input_size + hidden_size)
layer_2_weights_tensor = torch.stack(layer_2_gate_weights)  # Shape: (num_checkpoints, 4 gates, hidden_size, input_size + hidden_size)

# Save the tensors
torch.save(embedding_weights_tensor, "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/embedding_weights_fc.pt")
torch.save(output_weights_tensor, "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/output_weights_fc.pt")
torch.save(layer_1_weights_tensor, "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/layer_1_gate_weights_fc.pt")
torch.save(layer_2_weights_tensor, "/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/layer_2_gate_weights_fc.pt")

print("All weights saved successfully!")
