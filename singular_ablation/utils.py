import torch
import copy
from tqdm import tqdm
from src.language_models.model import RNNModel as lstm


def zero_out_singular_values(weight_matrix: torch.Tensor, n: int) -> torch.Tensor:
    # Compute SVD
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    
    # Zero out the n smallest singular values
    S[n]=0
    
    # Reconstruct the matrix
    modified_weight = (U * S.unsqueeze(0)) @ Vh
    return modified_weight


def modify_checkpoint_weight(checkpoint_path, layer, weight_type, gate, n, device):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_weights = checkpoint["model_state_dict"]
    
    # Extract original weight tensor (concatenated gates)
    original_weight = model_weights[f'rnn.weight_{weight_type}_l{layer}']  # shape: (4*gate_dim, dim)
    
    # Split into gates
    gates = dict(zip(['input', 'forget', 'cell', 'output'], original_weight.chunk(4, dim=0)))
    
    # Modify specified gate's weight matrix
    modified_gate_weight = zero_out_singular_values(gates[gate], n)
    
    # Replace gate weight in gates dict
    gates[gate] = modified_gate_weight
    
    # Re-concatenate gates into full weight matrix
    modified_weight = torch.cat([gates[g] for g in ['input', 'forget', 'cell', 'output']], dim=0)
    
    # Create a copy of checkpoint to avoid modifying the original
    new_checkpoint = copy.deepcopy(checkpoint)
    new_checkpoint["model_state_dict"][f'rnn.weight_{weight_type}_l{layer}'] = modified_weight
    
    return new_checkpoint

def evaluate_on_n_ablations(checkpoint_path, ntokens, layers, weight_type, gates, test_dataloader, n, init_sentence, device):
    ablation ={}
    for layer in layers:
        ablation[f'layer_{layer}'] = {}
        for gate in gates:
            
            ablation[f'layer_{layer}'][gate] = {}
            
            model = lstm('LSTM', ntokens, 650, 650, 2, 0, False).to(device)
            with open(checkpoint_path, "rb") as f:
                state_dict = torch.load(
                    f, map_location="cuda" if device == "cuda" else "cpu"
                )
                model.load_state_dict(state_dict["model_state_dict"])
                
            ppl_original = eval(model, test_dataloader, init_sentence)
            
            ablation[f'layer_{layer}'][gate]['original']=ppl_original
            ablation[f'layer_{layer}'][gate]['ablations'] = []
            
            for i in tqdm(range(n), desc=f"Layer {layer}, Gate {gate}"):
                check = modify_checkpoint_weight(checkpoint_path, layer, weight_type, gate, i, device)
                model = lstm('LSTM', ntokens, 650, 650, 2, 0, False).to(device)
                model.load_state_dict(check['model_state_dict'])
                ppl = eval(model, test_dataloader, init_sentence)
                ablation[f'layer_{layer}'][gate]['ablations'].append(ppl)
                
    return ablation   