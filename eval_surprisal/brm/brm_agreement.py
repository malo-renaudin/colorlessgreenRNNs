# These scripts are run with NYU greene high performance computing service
import sys
sys.path.append('../')
from util import load_data, Predicting_RT_with_spillover
from brms_parameters import get_brms_parameters
import bambi as bmb
import pandas as pd
import numpy as np
import pickle
import arviz as az

# Load data
rt_data = load_data("Agreement")
PredictedRT_df = Predicting_RT_with_spillover(rt_data, "Agreement")
PredictedRT_df = PredictedRT_df[PredictedRT_df['ROI'].isin([0, 1, 2])]

# Data transformations
PredictedRT_df.loc[PredictedRT_df['Type'] == "AGREE", 'Type'] = "AGREE_G"

# Split Type column
PredictedRT_df[['Type', 'pGram']] = PredictedRT_df['Type'].str.split('_', expand=True)

# Recode pGram values
PredictedRT_df.loc[PredictedRT_df['pGram'] == "UAMB", 'pGram'] = "G"
PredictedRT_df.loc[PredictedRT_df['pGram'] == "AMB", 'pGram'] = "U"
PredictedRT_df.loc[PredictedRT_df['pGram'] == "UNG", 'pGram'] = "U"

# Create coded variables
PredictedRT_df['pGram.coded'] = PredictedRT_df['pGram'].map({"U": 1, "G": 0})
PredictedRT_df['Type.coded'] = PredictedRT_df['Type'].map({"AGREE": 0, "NPZ": 1})

# Position coding
position_mapping_1 = {0: 0.5, 1: 0, 2: -0.5}
position_mapping_2 = {0: 0, 1: 0.5, 2: -0.5}
PredictedRT_df['position.coded.1'] = PredictedRT_df['ROI'].map(position_mapping_1)
PredictedRT_df['position.coded.2'] = PredictedRT_df['ROI'].map(position_mapping_2)

# Get BRMS parameters
brm_param_list = get_brms_parameters("prior1")

# Define priors (these would need to be adapted to bambi/PyMC syntax)
# Note: bambi uses different prior specification syntax than brms
# prior1 equivalent would be specified in bambi model definition

def fit_bambi_model(data, model_name, roi):
    """
    Fit a Bayesian model using bambi (Python equivalent of brms)
    """
    # Filter data
    filtered_data = data[
        (data['Type'] == "AGREE") & 
        (data['ROI'] == roi) & 
        (data['model'] == model_name) & 
        (data['RT'].notna())
    ].copy()
    
    # Define model formula
    # bambi syntax: outcome ~ predictors + (random_effects)
    formula = "predicted ~ pGram_coded + (1|item) + (1|participant) + (pGram_coded|item) + (pGram_coded|participant)"
    
    # Create model
    model = bmb.Model(formula, filtered_data)
    
    # Set priors (adapted to bambi syntax)
    priors = {
        "Intercept": bmb.Prior("Normal", mu=300, sigma=1000),
        "pGram_coded": bmb.Prior("Normal", mu=0, sigma=150),
        "1|item": bmb.Prior("Normal", mu=0, sigma=200),
        "1|participant": bmb.Prior("Normal", mu=0, sigma=200),
        "pGram_coded|item": bmb.Prior("Normal", mu=0, sigma=200),
        "pGram_coded|participant": bmb.Prior("Normal", mu=0, sigma=200),
        "sigma": bmb.Prior("Normal", mu=0, sigma=500)
    }
    
    # Fit model
    fitted_model = model.fit(
        draws=brm_param_list['niters'] - brm_param_list['warmup'],
        tune=brm_param_list['warmup'],
        chains=brm_param_list['ncores'],
        random_seed=brm_param_list['seed'],
        target_accept=brm_param_list['adapt_delta']
    )
    
    return fitted_model

# Rename column for bambi compatibility
PredictedRT_df['pGram_coded'] = PredictedRT_df['pGram.coded']

# LSTM models
print("Fitting LSTM models...")

# LSTM P0
brm_predicted_lstm_Agr_P0 = fit_bambi_model(PredictedRT_df, "lstm", 0)
with open("brm_predicted_lstm_Agr_P0.pkl", "wb") as f:
    pickle.dump(brm_predicted_lstm_Agr_P0, f)
print(az.summary(brm_predicted_lstm_Agr_P0))
del brm_predicted_lstm_Agr_P0

# LSTM P1
brm_predicted_lstm_Agr_P1 = fit_bambi_model(PredictedRT_df, "lstm", 1)
with open("brm_predicted_lstm_Agr_P1.pkl", "wb") as f:
    pickle.dump(brm_predicted_lstm_Agr_P1, f)
print(az.summary(brm_predicted_lstm_Agr_P1))
del brm_predicted_lstm_Agr_P1

# LSTM P2
brm_predicted_lstm_Agr_P2 = fit_bambi_model(PredictedRT_df, "lstm", 2)
with open("brm_predicted_lstm_Agr_P2.pkl", "wb") as f:
    pickle.dump(brm_predicted_lstm_Agr_P2, f)
print(az.summary(brm_predicted_lstm_Agr_P2))
del brm_predicted_lstm_Agr_P2

print("Fitting GPT-2 models...")

# GPT-2 models
# GPT-2 P0
brm_predicted_gpt2_Agr_P0 = fit_bambi_model(PredictedRT_df, "gpt2", 0)
with open("brm_predicted_gpt2_Agr_P0.pkl", "wb") as f:
    pickle.dump(brm_predicted_gpt2_Agr_P0, f)
print(az.summary(brm_predicted_gpt2_Agr_P0))
del brm_predicted_gpt2_Agr_P0

# GPT-2 P1
brm_predicted_gpt2_Agr_P1 = fit_bambi_model(PredictedRT_df, "gpt2", 1)
with open("brm_predicted_gpt2_Agr_P1.pkl", "wb") as f:
    pickle.dump(brm_predicted_gpt2_Agr_P1, f)
print(az.summary(brm_predicted_gpt2_Agr_P1))
del brm_predicted_gpt2_Agr_P1

# GPT-2 P2 (with custom iterations)
def fit_bambi_model_custom_iters(data, model_name, roi, draws=7500, tune=7500):
    """
    Fit a Bayesian model with custom iteration parameters
    """
    filtered_data = data[
        (data['Type'] == "AGREE") & 
        (data['ROI'] == roi) & 
        (data['model'] == model_name) & 
        (data['RT'].notna())
    ].copy()
    
    formula = "predicted ~ pGram_coded + (1|item) + (1|participant) + (pGram_coded|item) + (pGram_coded|participant)"
    model = bmb.Model(formula, filtered_data)
    
    fitted_model = model.fit(
        draws=draws,
        tune=tune,
        chains=brm_param_list['ncores'],
        random_seed=brm_param_list['seed'],
        target_accept=brm_param_list['adapt_delta']
    )
    
    return fitted_model

brm_predicted_gpt2_Agr_P2 = fit_bambi_model_custom_iters(PredictedRT_df, "gpt2", 2, draws=7500, tune=7500)
with open("brm_predicted_gpt2_Agr_P2.pkl", "wb") as f:
    pickle.dump(brm_predicted_gpt2_Agr_P2, f)
print(az.summary(brm_predicted_gpt2_Agr_P2))
del brm_predicted_gpt2_Agr_P2

print("All models fitted and saved successfully!")