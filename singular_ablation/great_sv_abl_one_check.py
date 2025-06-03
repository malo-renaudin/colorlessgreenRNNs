import sys
import os

sys.path.append('/scratch2/mrenaudin/colorlessgreenRNNs')

import torch
from src.language_models.dictionary_corpus import Dictionary, Corpus, tokenize
from src.language_models.model import RNNModel as lstm
from src.language_models.utils import move_to_device, batchify, get_batch, repackage_hidden
import torch.nn as nn
import math
from pathlib import Path
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from evaluation_notebooks.utils import NounPPDataset, BLiMPDataset, collate_fn_nounpp, collate_fn_blimp, feed_input, feed_sentence, eval_lstm_blimp, eval_lstm_nounpp
from utils import zero_out_singular_values, modify_checkpoint_weight
import logging
########################################################################################
# General Parameters and Necessary Files
########################################################################################

batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data"
dictionary = Dictionary(data_path)
nounpp = "//scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt"
layers = [0,1]
weight_type = 'hh'
gates = ['cell','forget','input','output']
ntokens = 50001
n = 10
checkpoint_path = '/scratch2/mrenaudin/colorlessgreenRNNs/checkpoints/lstm_adam_full_check_shuffled/epoch_40.pt'

########################################################################################
# Blimp Tasks and Init Sentence for Priming on NounPP
########################################################################################

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
               'drop_argument', 
               'ellipsis_n_bar_1', 
               'ellipsis_n_bar_2', 
               'existential_there_object_raising', 
               'existential_there_quantifiers_1', 
               'existential_there_quantifiers_2', 
               'existential_there_subject_raising', 
               'expletive_it_object_raising', 
               'inchoative', 
               'intransitive', 
               'irregular_past_participle_adjectives', 
               'irregular_past_participle_verbs', 
               'irregular_plural_subject_verb_agreement_1', 
               'irregular_plural_subject_verb_agreement_2', 
               'left_branch_island_echo_question', 
               'left_branch_island_simple_question', 
               'matrix_question_npi_licensor_present', 
               'npi_present_1', 
               'npi_present_2', 
               'only_npi_licensor_present', 
               'only_npi_scope', 
               'passive_1', 
               'passive_2', 
               'principle_A_c_command', 
               'principle_A_case_1', 
               'principle_A_case_2', 
               'principle_A_domain_1', 
               'principle_A_domain_2', 
               'principle_A_domain_3', 
               'principle_A_reconstruction', 
               'regular_plural_subject_verb_agreement_1', 
               'regular_plural_subject_verb_agreement_2', 
               'sentential_negation_npi_licensor_present', 
               'sentential_negation_npi_scope', 
               'sentential_subject_island', 
               'superlative_quantifiers_1', 
               'superlative_quantifiers_2', 
               'tough_vs_raising_1', 
               'tough_vs_raising_2', 
               'transitive', 
               'wh_island', 
               'wh_questions_object_gap', 
               'wh_questions_subject_gap', 
               'wh_questions_subject_gap_long_distance', 
               'wh_vs_that_no_gap', 
               'wh_vs_that_no_gap_long_distance', 
               'wh_vs_that_with_gap', 
               'wh_vs_that_with_gap_long_distance']

init_sentence = " ".join(
    [
        "In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
        'He even speculated that technical classes might some day be held " for the better training of workmen in their several crafts and industries . <eos>',
        "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
        'Moore says : " Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour " . <eos>',
        "<unk> is also the basis for online games sold through licensed lotteries . <eos>",
    ]
)
########################################################################################
# Load Model and Dataloaders
########################################################################################
nounpp = NounPPDataset(nounpp, dictionary)
nounpp_dataloader = DataLoader(nounpp, batch_size=batch_size, collate_fn=collate_fn_nounpp)

model = lstm('LSTM', ntokens, 650, 650, 2, 0, False).to(device)

with open(checkpoint_path, "rb") as f:
    state_dict = torch.load(
        f, map_location="cuda" if device == "cuda" else "cpu"
    )
    model.load_state_dict(state_dict["model_state_dict"])

########################################################################################
# Evaluation Function
########################################################################################
ablation ={}
ablation['original'] = {'Nounpp': None, 'Blimp': {}}

ppl_original_nounpp = eval_lstm_nounpp(model, nounpp_dataloader, init_sentence, dictionary, device)
ablation['original']['Nounpp']= ppl_original_nounpp

for task in blimp_tasks :
    task_dataset = BLiMPDataset(task, dictionary)
    task_dataloader = DataLoader(task_dataset, batch_size=526, collate_fn = collate_fn_blimp)
    ppl_original_blimp = eval_lstm_blimp(model, task_dataloader)
    ablation['original']['Blimp'][task]=ppl_original_blimp

for layer in layers:
    ablation[f'layer_{layer}'] = {}
    for gate in gates:
        ablation[f'layer_{layer}'][gate] = {'Nounpp': {}, 'Blimp': {}}
            
        for i in tqdm(range(n), desc=f"Layer {layer}, Gate {gate}"):
            
            #ablate singular value
            check = modify_checkpoint_weight(checkpoint_path, layer, weight_type, gate, i, device)
            model = lstm('LSTM', ntokens, 650, 650, 2, 0, False).to(device)
            model.load_state_dict(check['model_state_dict'])
            
            #evaluate nounpp
            ppl_nounpp = eval_lstm_nounpp(model, nounpp_dataloader, init_sentence, dictionary, device)
            ablation[f'layer_{layer}'][gate]['Nounpp'][str(i)] = ppl_nounpp
            
            #evaluate blimp
            for task in blimp_tasks:
                task_dataset = BLiMPDataset(task, dictionary)
                task_dataloader = DataLoader(task_dataset, batch_size=526, collate_fn = collate_fn_blimp)
                ppl_blimp = eval_lstm_blimp(model, task_dataloader)
                ablation[f'layer_{layer}'][gate]['Blimp'][task][str(i)] = ppl_blimp

torch.save(ablation, '/scratch2/mrenaudin/colorlessgreenRNNs/singular_ablation/great_sv_ablations_one_check.pt')