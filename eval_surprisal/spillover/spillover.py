"""
Spillover Analysis - Using statsmodels (No R required)

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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import warnings
from util import load_data
warnings.filterwarnings('ignore')

# Add path for util functions
# sys.path.append("../../analysis/shared/")
# from util import load_data

print(f"Analysis started: {datetime.now()}")



def load_surprisal_data():
    """Load SPR and surprisal data for fillers"""
    print("Loading data...")
    
    # Load SPR data
    spr = load_data("Fillers")
    
    # Load surprisal data from different models
    surps_lstm = pd.read_csv("eval_surprisal/get_surprisal/data/lstm/items_filler.lstm.csv.scaled")
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
    print(merged.columns)
    # Handle item column (assuming item.x exists after merge)
    if 'item_x' in merged.columns:
        merged['item'] = merged['item_x']
    
    # Use sum_surprisal_s (change to mean if more appropriate)
    merged['surprisal_s'] = merged['sum_surprisal_s']
    print(merged.columns)
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

def fit_filler_model_statsmodels(data, model_name):
    """
    Fit linear mixed effects model for filler data using statsmodels
    """
    print(f"Fitting {model_name} filler model with statsmodels...")
    
    # Scale WordPosition
    data['WordPosition_scaled'] = (data['WordPosition'] - data['WordPosition'].mean()) / data['WordPosition'].std()
    
    # Create interaction terms
    data['logfreq_length'] = data['logfreq_s'] * data['length_s']
    data['logfreq_p1_length_p1'] = data['logfreq_p1_s'] * data['length_p1_s']
    data['logfreq_p2_length_p2'] = data['logfreq_p2_s'] * data['length_p2_s']
    data['logfreq_p3_length_p3'] = data['logfreq_p3_s'] * data['length_p3_s']
    
    # Clean data - remove any remaining NaN values
    data_clean = data.dropna(subset=[
        'RT', 'surprisal_s', 'surprisal_p1_s', 'surprisal_p2_s', 'surprisal_p3_s',
        'WordPosition_scaled', 'logfreq_length', 'logfreq_p1_length_p1', 
        'logfreq_p2_length_p2', 'logfreq_p3_length_p3', 'participant', 'item'
    ])
    
    print(f"Cleaned dataset size: {len(data_clean)} rows")
    
    # Convert categorical variables to ensure proper handling
    data_clean['participant'] = data_clean['participant'].astype('category')
    data_clean['item'] = data_clean['item'].astype('category')
    
    # Define the model formula (simplified mixed effects using statsmodels)
    # Note: statsmodels has limited mixed effects support compared to R's lme4
    formula = """RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +
                     WordPosition_scaled + logfreq_length + logfreq_p1_length_p1 + 
                     logfreq_p2_length_p2 + logfreq_p3_length_p3"""
    
    try:
        # Try mixed linear model (requires groups to be specified)
        model = smf.mixedlm(formula, data_clean, groups=data_clean['participant'], 
                           re_formula="~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s")
        result = model.fit(method='lbfgs')
        
    except Exception as e:
        print(f"Mixed model failed: {e}")
        print("Falling back to OLS with participant/item fixed effects...")
        
        # Fallback to OLS with fixed effects
        # Add participant and item dummy variables
        participant_dummies = pd.get_dummies(data_clean['participant'], prefix='participant')
        item_dummies = pd.get_dummies(data_clean['item'], prefix='item')
        
        # Combine all predictors
        X_vars = ['surprisal_s', 'surprisal_p1_s', 'surprisal_p2_s', 'surprisal_p3_s',
                  'WordPosition_scaled', 'logfreq_length', 'logfreq_p1_length_p1', 
                  'logfreq_p2_length_p2', 'logfreq_p3_length_p3']
        
        model_data = pd.concat([
            data_clean[['RT'] + X_vars],
            participant_dummies.iloc[:, :-1],  # Drop one dummy to avoid collinearity
            item_dummies.iloc[:, :-1]          # Drop one dummy to avoid collinearity
        ], axis=1)
        
        # Fit OLS model
        X = model_data.drop('RT', axis=1)
        y = model_data['RT']
        
        X = sm.add_constant(X)  # Add intercept
        result = sm.OLS(y, X).fit()
    
    return result

def fit_simple_regression_model(data, model_name):
    """
    Fit a simplified regression model focusing on surprisal effects
    """
    print(f"Fitting simplified {model_name} model...")
    
    # Scale WordPosition
    data['WordPosition_scaled'] = (data['WordPosition'] - data['WordPosition'].mean()) / data['WordPosition'].std()
    
    # Create interaction terms
    data['logfreq_length'] = data['logfreq_s'] * data['length_s']
    data['logfreq_p1_length_p1'] = data['logfreq_p1_s'] * data['length_p1_s']
    data['logfreq_p2_length_p2'] = data['logfreq_p2_s'] * data['length_p2_s']
    data['logfreq_p3_length_p3'] = data['logfreq_p3_s'] * data['length_p3_s']
    
    # Clean data
    data_clean = data.dropna(subset=[
        'RT', 'surprisal_s', 'surprisal_p1_s', 'surprisal_p2_s', 'surprisal_p3_s',
        'WordPosition_scaled', 'logfreq_length', 'logfreq_p1_length_p1', 
        'logfreq_p2_length_p2', 'logfreq_p3_length_p3'
    ])
    
    print(f"Cleaned dataset size: {len(data_clean)} rows")
    
    # Simple OLS regression focusing on the key effects
    formula = """RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +
                     WordPosition_scaled + logfreq_length + logfreq_p1_length_p1 + 
                     logfreq_p2_length_p2 + logfreq_p3_length_p3"""
    
    result = smf.ols(formula, data=data_clean).fit()
    
    return result

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
    
    # Try mixed effects model first, fallback to simpler model if needed
    try:
        models_filler_lstm = fit_filler_model_statsmodels(dropped_lstm, "LSTM")
    except Exception as e:
        print(f"Advanced model failed: {e}")
        print("Using simplified regression model...")
        models_filler_lstm = fit_simple_regression_model(dropped_lstm, "LSTM")
    
    print(models_filler_lstm.summary())
    
    # Save model
    with open("eval_surprisal/filler_models/filler_lstm_sum.pkl", "wb") as f:
        pickle.dump(models_filler_lstm, f)
    
    # Save model coefficients and summary statistics
    model_info = {
        'model': models_filler_lstm,
        'params': models_filler_lstm.params,
        'pvalues': models_filler_lstm.pvalues,
        'conf_int': models_filler_lstm.conf_int(),
        'summary': str(models_filler_lstm.summary())
    }
    
    with open("filler_models/filler_lstm_sum_info.pkl", "wb") as f:
        pickle.dump(model_info, f)
    
    # Free memory
    del dropped_lstm
    gc.collect()
    
    print(f"\nAnalysis completed: {datetime.now()}")
    print("LSTM filler model saved successfully!")
    print(f"Model coefficients for surprisal effects:")
    surprisal_params = [param for param in models_filler_lstm.params.index 
                       if 'surprisal' in param]
    for param in surprisal_params:
        print(f"  {param}: {models_filler_lstm.params[param]:.4f} (p={models_filler_lstm.pvalues[param]:.4f})")

if __name__ == "__main__":
    main()