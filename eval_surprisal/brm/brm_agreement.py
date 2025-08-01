import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level from the script's directory (evaluation_notebooks)
parent_dir = os.path.dirname(script_dir)
grand_parent_dir = os.path.dirname(parent_dir)
# Join with 'src' to get the correct path to the src directory
#src_dir = os.path.join(grand_parent_dir, "src")
print(grand_parent_dir)
sys.path.append(os.path.abspath(grand_parent_dir))
from eval_surprisal.util import load_data, Predicting_RT_with_spillover, process_model_data
from eval_surprisal.brm.brm_parameters import get_brms_parameters
import bambi as bmb
import pandas as pd
import numpy as np
import pickle
import arviz as az
import pyreadr  # For loading .rds files
print('ok')

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

# Rename column for bambi compatibility
PredictedRT_df['pGram_coded'] = PredictedRT_df['pGram.coded']

# Drop predicted column if it exists
if 'predicted' in PredictedRT_df.columns:
    PredictedRT_df = PredictedRT_df.drop(columns='predicted')

def load_pretrained_model(filepath, model_type="rds"):
    """
    Load a pretrained model from different formats
    
    Parameters:
    -----------
    filepath : str
        Path to the model file
    model_type : str
        Type of model file ("rds", "pkl", "nc")
    
    Returns:
    --------
    object
        Loaded model object
    """
    try:
        if model_type == "rds":
            # Load R model using pyreadr
            print(f"Loading RDS model from {filepath}")
            result = pyreadr.read_r(filepath)
            model = result[None]  # RDS files contain single object
            print(f"Successfully loaded RDS model, type: {type(model)}")
            return model
            
        elif model_type == "pkl":
            # Load Python pickle file
            print(f"Loading pickle model from {filepath}")
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Successfully loaded pickle model, type: {type(model)}")
            return model
            
        elif model_type == "nc":
            # Load NetCDF file (ArviZ format)
            print(f"Loading NetCDF model from {filepath}")
            model = az.from_netcdf(filepath)
            print(f"Successfully loaded NetCDF model, type: {type(model)}")
            return model
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None

def analyze_pretrained_model(model, model_name):
    """
    Analyze a pretrained model and extract key information
    
    Parameters:
    -----------
    model : object
        Loaded model object
    model_name : str
        Name of the model for identification
    """
    print(f"\n=== Analysis for {model_name} ===")
    
    try:
        # Try to get model summary if it's an ArviZ InferenceData object
        if hasattr(model, 'posterior'):
            print("Model contains posterior samples")
            summary = az.summary(model)
            print(summary)
            
            # Plot diagnostics
            try:
                az.plot_trace(model)
                az.plot_posterior(model)
            except Exception as e:
                print(f"Could not create plots: {e}")
                
        # If it's an R brms model, try to extract information
        elif hasattr(model, 'names'):
            print(f"R model components: {list(model.names)}")
            
            # Try to access common brms components
            if 'fit' in model.names:
                print("Model contains Stan fit object")
            if 'data' in model.names:
                print("Model contains data")
            if 'formula' in model.names:
                print(f"Model formula: {model.rx2('formula')}")
                
        else:
            print(f"Unknown model structure. Available attributes: {dir(model)}")
            
    except Exception as e:
        print(f"Error analyzing model: {e}")

def fit_bambi_model(data, model_name, roi):
    """
    Fit a Bayesian model using bambi (only if pretrained model not available)
    """
    # Filter data
    filtered_data = data[
        (data['Type'] == "AGREE") & 
        (data['ROI'] == roi) & 
        (data['model'] == model_name) & 
        (data['RT'].notna())
    ].copy()
    
    # Define model formula
    formula = "RT ~ pGram_coded + (1|item) + (1|participant) + (pGram_coded|item) + (pGram_coded|participant)"
    
    print("Missing values per column:")
    print(filtered_data.isnull().sum())
    print(f"\nTotal rows: {len(filtered_data)}")
    print(f"Complete rows: {len(filtered_data.dropna())}")
    
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

# Configuration: Set USE_PRETRAINED to True to load existing models
USE_PRETRAINED = True  # Change to False if you want to retrain
PRETRAINED_DIR = "eval_surprisal/pretrained_models/agreement"  # Directory containing pretrained models

# Model configurations
models_config = [
    {"name": "brm_predicted_lstm_Agr_P0", "model_type": "lstm", "roi": 0},
    {"name": "brm_predicted_lstm_Agr_P1", "model_type": "lstm", "roi": 1}, 
    {"name": "brm_predicted_lstm_Agr_P2", "model_type": "lstm", "roi": 2}
]

# Process models
loaded_models = {}

for config in models_config:
    model_name = config["name"]
    model_type = config["model_type"] 
    roi = config["roi"]
    
    if USE_PRETRAINED:
        # Try to load pretrained model (check multiple formats)
        model_loaded = False
        
        # Try .rds first (most likely for brms models)
        rds_path = os.path.join(PRETRAINED_DIR, f"{model_name}.rds")
        if os.path.exists(rds_path):
            model = load_pretrained_model(rds_path, "rds")
            if model is not None:
                loaded_models[model_name] = model
                analyze_pretrained_model(model, model_name)
                model_loaded = True
        
        # Try .pkl if .rds not found
        if not model_loaded:
            pkl_path = os.path.join(PRETRAINED_DIR, f"{model_name}.pkl")
            if os.path.exists(pkl_path):
                model = load_pretrained_model(pkl_path, "pkl")
                if model is not None:
                    loaded_models[model_name] = model
                    analyze_pretrained_model(model, model_name)
                    model_loaded = True
        
        # Try .nc if others not found
        if not model_loaded:
            nc_path = os.path.join(PRETRAINED_DIR, f"{model_name}.nc")
            if os.path.exists(nc_path):
                model = load_pretrained_model(nc_path, "nc")
                if model is not None:
                    loaded_models[model_name] = model
                    analyze_pretrained_model(model, model_name)
                    model_loaded = True
        
        if not model_loaded:
            print(f"No pretrained model found for {model_name}, will train new model")
            USE_PRETRAINED = False
    
    if not USE_PRETRAINED:
        # Train new model
        print(f"Training new model: {model_name}")
        fitted_model = fit_bambi_model(PredictedRT_df, model_type, roi)
        
        # Save the newly trained model
        with open(f"{model_name}.pkl", "wb") as f:
            pickle.dump(fitted_model, f)
        
        loaded_models[model_name] = fitted_model
        print(az.summary(fitted_model))
        
        # Clean up memory
        del fitted_model

print(f"\nSuccessfully processed {len(loaded_models)} models:")
for name in loaded_models.keys():
    print(f"  - {name}")

# Optional: Compare models if multiple are loaded
if len(loaded_models) > 1:
    try:
        # This only works if models are ArviZ InferenceData objects
        inference_data_models = {k: v for k, v in loaded_models.items() 
                               if hasattr(v, 'posterior')}
        if len(inference_data_models) > 1:
            comparison = az.compare(inference_data_models)
            print("\nModel Comparison:")
            print(comparison)
    except Exception as e:
        print(f"Could not compare models: {e}")

print("Analysis completed!")