import sys
import os

sys.path.append('colorlessgreenRNNs')

from src.language_models import model as m
import torch
from evaluation_notebooks.utils import BLiMPDataset, collate_fn_blimp
from src.language_models.dictionary_corpus import Dictionary
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def compute_seq_nll(data, model, cache, nheads, temperature, gumbel):
    seq_len, batch_size = data.shape
    mask = (data!=0).float()

    output, cache = model(data, cache, nheads, temperature, gumbel)
    targets = data[1:, :]

    log_probs = F.log_softmax(output, dim=-1)
    log_probs = log_probs[:-1]
    log_probs = log_probs.swapaxes(0,1)

    nll_loss = F.nll_loss(
            log_probs.reshape(-1, log_probs.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )
    
    nll_loss=nll_loss.reshape(seq_len - 1, batch_size)
    masked_nll_loss = nll_loss * mask[1:, :]
    sequence_nll = masked_nll_loss.sum(dim=1)
    return -sequence_nll
 
def eval_one_task(model, 
                  data_path, 
                  blimp_task, 
                  batch_size, 
                  gumbel, 
                  nheads, 
                  temperature, 
                  device
                  ):
    
    
    dictionary = Dictionary(data_path)
    blimp = BLiMPDataset(blimp_task, dictionary)
    test_dataloader = DataLoader(blimp, batch_size, collate_fn = collate_fn_blimp)
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_dataloader:

            good = batch['encoded_good'].transpose(0,1).to(device)
            bad = batch['encoded_bad'].transpose(0,1).to(device)
            batch_size = good.size(0)
            
            cache_good = model.init_cache(good, 1)
            seq_nll_good = compute_seq_nll(good, model, cache_good, nheads, temperature, gumbel)
            cache_bad = model.init_cache(bad, 1)
            seq_nll_bad = compute_seq_nll(bad, model, cache_bad, nheads, temperature, gumbel)
            min_len = min(seq_nll_good.size(0), seq_nll_bad.size(0))
            seq_nll_good = seq_nll_good[:min_len]
            seq_nll_bad = seq_nll_bad[:min_len]
            predictions = (seq_nll_good > seq_nll_bad).cpu().numpy()
            
            correct_predictions += np.sum(predictions)
            total_predictions += batch_size
                
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on {test_dataloader.dataset.dataset.config_name}: {accuracy * 100:.2f}%")
    return accuracy

def eval_all_blimp(checkpoint, 
                  data_path, 
                  gumbel, 
                  nheads,
                  temperature,
                  hidden_dim, 
                  device):
    
    model = m.CBR_RNN(50001, hidden_dim, hidden_dim, nheads, 0.5, device)
    model=model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    checkpoint_results = {}
    
    blimp_tasks = ['adjunct_island', 
                   'anaphor_gender_agreement', 
                   'anaphor_number_agreement', 
                   'animate_subject_passive', 
                   'animate_subject_trans', 
                   'causative', 
                   'complex_NP_island', 
                   'coordinate_structure_constraint_complex_left_branch', 
                   'coordinate_structure_constraint_object_extraction', 
                   'determiner_noun_agreement_1', 
                   'determiner_noun_agreement_2', 
                   'determiner_noun_agreement_irregular_1', 
                   'determiner_noun_agreement_irregular_2', 
                   'determiner_noun_agreement_with_adj_2', 
                   'determiner_noun_agreement_with_adj_irregular_1', 
                   'determiner_noun_agreement_with_adj_irregular_2', 
                   'determiner_noun_agreement_with_adjective_1', 
                   'distractor_agreement_relational_noun', 
                   'distractor_agreement_relative_clause', 
                   'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 
                   'existential_there_object_raising', 'existential_there_quantifiers_1', 
                   'existential_there_quantifiers_2', 'existential_there_subject_raising', 
                   'expletive_it_object_raising', 'inchoative', 'intransitive', 
                   'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 
                   'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 
                   'left_branch_island_echo_question', 'left_branch_island_simple_question', 
                   'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 
                   'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 
                   'passive_2', 'principle_A_c_command', 'principle_A_case_1', 
                   'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 
                   'principle_A_domain_3', 'principle_A_reconstruction', 
                   'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 
                   'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 
                   'sentential_subject_island', 'superlative_quantifiers_1', 
                   'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 
                   'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 
                   'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 
                   'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']
    
    for task in blimp_tasks:
        checkpoint_results[task]= eval_one_task(model, 
                                                data_path, 
                                                task, 
                                                512, 
                                                gumbel, 
                                                nheads, 
                                                temperature,
                                                device
                                                )
    return checkpoint_results
        
