"""
Spillover Analysis

This document will fit the *filler model*, a linear mixed effects model mapping surprisals 0--3 words back 
(and several nuisance factors) to reading times over the filler items. This model will be used to convert 
surprisals to reading times in the analysis code in each subset's analysis code.
"""

import pandas as pd
import numpy as np
import sys
import os
import gc
import pickle
from datetime import datetime
import pymer4
from pymer4.models import Lmer
import warnings
warnings.filterwarnings('ignore')

# Add path for util functions
sys.path.append("../../analysis/shared/")
from util import load_data

print(f"Analysis started: {datetime.now()}")

def load_surprisal_data():
    """Load SPR and surprisal data for fillers"""
    print("Loading data...")
    
    # Load SPR data
    spr = load_data("Fillers")
    
    # Load surprisal data from different models
    surps_lstm = pd.read_csv("../data/lstm/items_filler.lstm.csv.scaled")
    # surps_gpt2 = pd.read_csv("../data/gpt2/items_filler.gpt2.csv.scaled")
    # surps_rnng = pd.read_csv("../data/rnng/items_filler.rnng.csv.scaled")
    
    # Adjust to 1-indexing
    surps_lstm['word_pos'] = surps_lstm['word_pos'] + 1
    # surps_gpt2['word_pos'] = surps_gpt2['word_pos'] + 1
    # surps_rnng['word_pos'] = surps_rnng['word_pos'] + 1
    
    return spr, surps_lstm#, surps_gpt2, surps_rnng

def bind_surps(spr, surps):
    """
    Merge SPR data with surprisal data and create lagged variables for spillover effects
    """
    # Merge datasets
    merged = pd.merge(spr, surps, 
                     left_on=['Sentence', 'WordPosition'], 
                     right_on=['Sentence', 'word_pos'], 
                     how='left')
    
    # Handle item column (assuming item.x exists after merge)
    if 'item.x' in merged.columns:
        merged['item'] = merged['item.x']
    
    # Use sum_surprisal_s (change to mean if more appropriate)
    merged['surprisal_s'] = merged['sum_surprisal_s']
    
    # Create lagged variables for spillover effects
    # Group by item and participant to ensure lags are within the same context
    grouped = merged.groupby(['item', 'participant'])
    
    # Create lag variables
    merged['RT_p1'] = grouped['RT'].shift(1)
    merged['RT_p2'] = grouped['RT'].shift(2) 
    merged['RT_p3'] = grouped['RT'].shift(3)
    
    merged['length_p1_s'] = grouped['length_s'].shift(1)
    merged['length_p2_s'] = grouped['length_s'].shift(2)
    merged['length_p3_s'] = grouped['length_s'].shift(3)
    
    merged['logfreq_p1_s'] = grouped['logfreq_s'].shift(1)
    merged['logfreq_p2_s'] = grouped['logfreq_s'].shift(2)
    merged['logfreq_p3_s'] = grouped['logfreq_s'].shift(3)
    
    merged['surprisal_p1_s'] = grouped['surprisal_s'].shift(1)
    merged['surprisal_p2_s'] = grouped['surprisal_s'].shift(2)
    merged['surprisal_p3_s'] = grouped['surprisal_s'].shift(3)
    
    # Calculate sentence length
    merged['sent_length'] = merged['Sentence'].str.split().str.len()
    
    # Filter out rows with missing data and last words of sentences
    dropped = merged[
        merged['surprisal_s'].notna() &
        merged['surprisal_p1_s'].notna() & 
        merged['surprisal_p2_s'].notna() &
        merged['surprisal_p3_s'].notna() &
        merged['logfreq_s'].notna() & 
        merged['logfreq_p1_s'].notna() &
        merged['logfreq_p2_s'].notna() & 
        merged['logfreq_p3_s'].notna() & 
        (merged['sent_length'] != merged['WordPosition'])
    ].copy()
    
    print(f"Dropped: {len(merged) - len(dropped)} rows")
    return dropped

def fit_filler_model(data, model_name):
    """
    Fit linear mixed effects model for filler data
    """
    print(f"Fitting {model_name} filler model...")
    
    # Scale WordPosition
    data['WordPosition_scaled'] = (data['WordPosition'] - data['WordPosition'].mean()) / data['WordPosition'].std()
    
    # Create interaction terms
    data['logfreq_length'] = data['logfreq_s'] * data['length_s']
    data['logfreq_p1_length_p1'] = data['logfreq_p1_s'] * data['length_p1_s']
    data['logfreq_p2_length_p2'] = data['logfreq_p2_s'] * data['length_p2_s']
    data['logfreq_p3_length_p3'] = data['logfreq_p3_s'] * data['length_p3_s']
    
    # Define the model formula
    formula = """RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +
                     WordPosition_scaled + logfreq_length + logfreq_p1_length_p1 + 
                     logfreq_p2_length_p2 + logfreq_p3_length_p3 + 
                     (1|participant) + (surprisal_s|participant) + (surprisal_p1_s|participant) + 
                     (surprisal_p2_s|participant) + (surprisal_p3_s|participant) + (1|item)"""
    
    # Fit model using pymer4
    model = Lmer(formula, data=data)
    
    # Fit with optimization settings similar to R's bobyqa
    try:
        model.fit(
            optimize='L-BFGS-B',
            control={'maxfun': 200000},
            verbose=True
        )
    except Exception as e:
        print(f"Model fitting failed with L-BFGS-B, trying SLSQP: {e}")
        try:
            model.fit(
                optimize='SLSQP',
                control={'maxiter': 10000},
                verbose=True
            )
        except Exception as e2:
            print(f"Model fitting failed with SLSQP: {e2}")
            raise
    
    return model

def main():
    """Main analysis pipeline"""
    # Load data
    spr, surps_lstm = load_surprisal_data()
    
    # Create output directory
    os.makedirs("filler_models", exist_ok=True)
    
    # Process LSTM model
    print("\n" + "="*50)
    print("Processing LSTM model")
    print("="*50)
    
    dropped_lstm = bind_surps(spr, surps_lstm)
    models_filler_lstm = fit_filler_model(dropped_lstm, "LSTM")
    print(models_filler_lstm)
    
    # Save model
    with open("filler_models/filler_lstm_sum.pkl", "wb") as f:
        pickle.dump(models_filler_lstm, f)
    
    # Free memory
    del dropped_lstm
    gc.collect()
    
    # # Process GPT-2 model
    # print("\n" + "="*50)
    # print("Processing GPT-2 model")
    # print("="*50)
    
    # dropped_gpt2 = bind_surps(spr, surps_gpt2)
    # models_filler_gpt2 = fit_filler_model(dropped_gpt2, "GPT-2")
    # print(models_filler_gpt2)
    
    # # Save model
    # with open("filler_models/filler_gpt2_sum.pkl", "wb") as f:
    #     pickle.dump(models_filler_gpt2, f)
    
    # # Free memory
    # del dropped_gpt2
    # gc.collect()
    
    # # Process RNNG model  
    # print("\n" + "="*50)
    # print("Processing RNNG model")
    # print("="*50)
    
    # dropped_rnng = bind_surps(spr, surps_rnng)
    # models_filler_rnng = fit_filler_model(dropped_rnng, "RNNG")
    # print(models_filler_rnng)
    
    # # Save model
    # with open("filler_models/filler_rnng_sum.pkl", "wb") as f:
    #     pickle.dump(models_filler_rnng, f)
    
    # # Free memory
    # del dropped_rnng
    # gc.collect()
    
    print(f"\nAnalysis completed: {datetime.now()}")
    print("All filler models saved successfully!")

if __name__ == "__main__":
    main()