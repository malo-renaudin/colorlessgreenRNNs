import sys
import pandas as pd
from pandas.core.indexes import base, range as pd_range
import types

# Create the missing numeric module mapping
numeric_module = types.ModuleType('pandas.core.indexes.numeric')

# Copy everything from base module
for attr_name in dir(base):
    if not attr_name.startswith('_'):
        setattr(numeric_module, attr_name, getattr(base, attr_name))

# Add range-specific items
for attr_name in dir(pd_range):
    if not attr_name.startswith('_'):
        setattr(numeric_module, attr_name, getattr(pd_range, attr_name))

# Register the module
sys.modules['pandas.core.indexes.numeric'] = numeric_module

print("Successfully created pandas.core.indexes.numeric module mapping")

import pickle
import numpy as np 
import sys
import joblib
import sys
import pandas as pd

# sys.modules['pandas.core.indexes.numeric'] = numeric

# Print current pandas version and available modules
print(f"Pandas version: {pd.__version__}")
print("Available pandas.core.indexes modules:")
import pandas.core.indexes
print(dir(pandas.core.indexes))

def load_data(subsetname, RTcutoffhigh=7000, RTcutofflow=0):
    # Map subset names to filenames
    files = {
        "ClassicGP": "ClassicGardenPathSet.csv",
        "RelativeClause": "RelativeClauseSet.csv", 
        "AttachmentAmbiguity": "AttachmentSet.csv",
        "Agreement": "AgreementSet.csv",
        "Fillers": "Fillers.csv"
    }
    
    # Load CSV
    root_path = '/scratch2/mrenaudin/colorlessgreenRNNs/eval_surprisal/spr_data'
    path = f'{root_path}/{files[subsetname]}'
    rt_data = pd.read_csv(path)
    
    # Create participant column
    rt_data['participant'] = rt_data['MD5']
    
    # Filter RTs
    rt_data.loc[(rt_data['RT'] > RTcutoffhigh) | (rt_data['RT'] < RTcutofflow), 'RT'] = np.nan
    
    # Clean text
    rt_data['Sentence'] = rt_data['Sentence'].str.replace('%2C', ',')
    rt_data['EachWord'] = rt_data['EachWord'].str.replace('%2C', ',')
    
    # Create word column (lowercase, remove trailing punctuation)
    rt_data['word'] = rt_data['EachWord'].str.replace(r'[.,]$', '', regex=True).str.lower()
    
    return rt_data


def Predicting_RT_with_spillover(rt_data_df, subsetname, models=['lstm']):
    """
    Predict reading times with spillover effects using trained filler models
    
    Parameters:
    -----------
    rt_data_df : pd.DataFrame
        Reading time data
    subsetname : str
        Name of the subset ('Agreement', 'ClassicGP', etc.)
    models : list
        List of models to process ('gpt2', 'lstm', 'nosurp')
    
    Returns:
    --------
    pd.DataFrame
        Combined data with predictions from all models
    """
    print("This will take a while.")
    
    pred_list = []
    
    if subsetname == "Agreement":
        i = 0
        for model in models:
            if i != 2:  # Not the nosurp model (third model)
                print(f'Processing model {model}')
                
                # Load surprisal data for Agreement
                surps_path = f'eval_surprisal/get_surprisal/data/{model}/items_{subsetname}.{model}.csv.scaled'
                surps = pd.read_csv(surps_path)
                surps['word_pos'] = surps['word_pos'] + 1  # Adjust to 1-indexing
                surps['model'] = model
                surps = surps[['Sentence', 'word_pos', 'sum_surprisal', 'sum_surprisal_s', 
                              'logfreq', 'logfreq_s', 'length', 'length_s']]
                surps = surps.rename(columns={'sum_surprisal': 'surprisal', 
                                            'sum_surprisal_s': 'surprisal_s'})
                
                # Load ClassicGP data for NPZ conditions
                surps2_path = f'eval_surprisal/get_surprisal/data/{model}/items_ClassicGP.{model}.csv.scaled'
                surps2 = pd.read_csv(surps2_path)
                
                # Filter for NPZ conditions and matching items
                agree_items = rt_data_df[rt_data_df['Type'] == 'AGREE']['item'].unique()
                surps2 = surps2[
                    surps2['condition'].isin(['NPZ_UAMB', 'NPZ_AMB']) & 
                    surps2['item'].isin(agree_items)
                ]
                
                surps2['word_pos'] = surps2['word_pos'] + 1  # Adjust to 1-indexing
                surps2['model'] = model
                surps2 = surps2[['Sentence', 'word_pos', 'sum_surprisal', 'sum_surprisal_s', 
                                'logfreq', 'logfreq_s', 'length', 'length_s']]
                surps2 = surps2.rename(columns={'sum_surprisal': 'surprisal', 
                                              'sum_surprisal_s': 'surprisal_s'})
                
                # Combine surprisal data
                surps = pd.concat([surps, surps2], ignore_index=True)
                
                # Load filler model
                filler_model_path = f'eval_surprisal/filler_models/filler_{model}_sum.pkl'
                with open(filler_model_path, 'rb') as f:
                    filler_model = joblib.load(f)
                
            else:  # nosurp model (third model)
                print(f'Processing model {model}')
                
                # For nosurp model, use LSTM data but nosurp filler model
                surps_path = f'eval_surprisal/get_surprisal/data/lstm/items_{subsetname}.lstm.csv.scaled'
                surps = pd.read_csv(surps_path)
                surps['word_pos'] = surps['word_pos'] + 1  # Adjust to 1-indexing
                surps['model'] = model
                surps = surps[['Sentence', 'word_pos', 'sum_surprisal', 'sum_surprisal_s', 
                              'logfreq', 'logfreq_s', 'length', 'length_s']]
                surps = surps.rename(columns={'sum_surprisal': 'surprisal', 
                                            'sum_surprisal_s': 'surprisal_s'})
                
                # Load ClassicGP data for NPZ conditions
                surps2_path = f'eval_surprisal/get_surprisal/data/lstm/items_ClassicGP.lstm.csv.scaled'
                surps2 = pd.read_csv(surps2_path)
                
                agree_items = rt_data_df[rt_data_df['Type'] == 'AGREE']['item'].unique()
                surps2 = surps2[
                    surps2['condition'].isin(['NPZ_UAMB', 'NPZ_AMB']) & 
                    surps2['item'].isin(agree_items)
                ]
                
                surps2['word_pos'] = surps2['word_pos'] + 1
                surps2['model'] = model
                surps2 = surps2[['Sentence', 'word_pos', 'sum_surprisal', 'sum_surprisal_s', 
                                'logfreq', 'logfreq_s', 'length', 'length_s']]
                surps2 = surps2.rename(columns={'sum_surprisal': 'surprisal', 
                                              'sum_surprisal_s': 'surprisal_s'})
                
                surps = pd.concat([surps, surps2], ignore_index=True)
                
                # Load nosurp filler model
                filler_model_path = f'eval_surprisal/filler_models/filler_sum_nosurp.pkl'
                with open(filler_model_path, 'rb') as f:
                    try:
                        filler_model = pickle.load(f)
                    except:
                        f.seek(0)
                        filler_model = pickle.load(f, encoding='latin1')
                            
            # Process data (same for both cases)
            rt_data_processed = process_model_data(rt_data_df, surps, filler_model, model)
            pred_list.append(rt_data_processed)
            i += 1
    
    else:  # For other subsets (not Agreement)
        i = 0
        for model in models:
            if i != 2:  # Not the nosurp model
                print(f'Processing model {model}')
                
                # Load surprisal data
                surps_path = f'eval_surprisal/get_surprisal/data/{model}/items_{subsetname}.{model}.csv.scaled'
                surps = pd.read_csv(surps_path)
                surps['word_pos'] = surps['word_pos'] + 1  # Adjust to 1-indexing
                surps['model'] = model
                surps = surps[['Sentence', 'word_pos', 'sum_surprisal', 'sum_surprisal_s', 
                              'logfreq', 'logfreq_s', 'length', 'length_s']]
                surps = surps.rename(columns={'sum_surprisal': 'surprisal', 
                                            'sum_surprisal_s': 'surprisal_s'})
                
                # Load filler model
                filler_model_path = f'eval_surprisal/get_surprisal/data/filler_models/filler_{model}_sum.pkl'
                with open(filler_model_path, 'rb') as f:
                    try:
                        filler_model = pickle.load(f)
                    except:
                        f.seek(0)
                        filler_model = pickle.load(f, encoding='latin1')
                
            else:  # nosurp model
                print(f'Processing model {model}')
                
                # For nosurp model, use LSTM data
                surps_path = f'eval_surprisal/get_surprisal/data/lstm/items_{subsetname}.lstm.csv.scaled'
                surps = pd.read_csv(surps_path)
                surps['word_pos'] = surps['word_pos'] + 1  # Adjust to 1-indexing
                surps['model'] = model
                surps = surps[['Sentence', 'word_pos', 'sum_surprisal', 'sum_surprisal_s', 
                              'logfreq', 'logfreq_s', 'length', 'length_s']]
                surps = surps.rename(columns={'sum_surprisal': 'surprisal', 
                                            'sum_surprisal_s': 'surprisal_s'})
                
                # Load nosurp filler model
                filler_model_path = f'eval_surprisal/get_surprisal/data/filler_models/filler_sum_nosurp.pkl'
                with open(filler_model_path, 'rb') as f:
                    filler_model = joblib.load(f)
            
            # Process data
            rt_data_processed = process_model_data(rt_data_df, surps, filler_model, model)
            pred_list.append(rt_data_processed)
            i += 1
    
    # Combine all predictions
    pred_dat = pd.concat(pred_list, ignore_index=True)
    
    return pred_dat


def process_model_data(rt_data_df, surps, filler_model, model):
    """
    Process RT data with surprisal data and make predictions
    
    Parameters:
    -----------
    rt_data_df : pd.DataFrame
        Reading time data
    surps : pd.DataFrame  
        Surprisal data
    filler_model : object
        Trained filler model
    model : str
        Model name
    
    Returns:
    --------
    pd.DataFrame
        Processed data with predictions
    """
    
    # Merge RT data with surprisal data
    rt_data_freqs_surps = pd.merge(
        rt_data_df, surps,
        left_on=['Sentence', 'WordPosition'],
        right_on=['Sentence', 'word_pos'],
        how='left'
    ).sort_values(['Type', 'item', 'WordPosition', 'participant'])
    
    # Create lagged variables for spillover effects
    grouped = rt_data_freqs_surps.groupby(['item', 'participant'])
    
    rt_data_freqs_surps['RT_p1'] = grouped['RT'].shift(1)
    rt_data_freqs_surps['RT_p2'] = grouped['RT'].shift(2)
    rt_data_freqs_surps['RT_p3'] = grouped['RT'].shift(3)
    
    rt_data_freqs_surps['length_p1_s'] = grouped['length_s'].shift(1)
    rt_data_freqs_surps['length_p2_s'] = grouped['length_s'].shift(2)
    rt_data_freqs_surps['length_p3_s'] = grouped['length_s'].shift(3)
    
    rt_data_freqs_surps['logfreq_p1_s'] = grouped['logfreq_s'].shift(1)
    rt_data_freqs_surps['logfreq_p2_s'] = grouped['logfreq_s'].shift(2)
    rt_data_freqs_surps['logfreq_p3_s'] = grouped['logfreq_s'].shift(3)
    
    rt_data_freqs_surps['surprisal_p1_s'] = grouped['surprisal_s'].shift(1)
    rt_data_freqs_surps['surprisal_p2_s'] = grouped['surprisal_s'].shift(2)
    rt_data_freqs_surps['surprisal_p3_s'] = grouped['surprisal_s'].shift(3)
    
    # Calculate sentence length
    rt_data_freqs_surps['sent_length'] = rt_data_freqs_surps['Sentence'].str.split().str.len()
    
    # Filter out rows with missing data and last words of sentences
    rt_data_drop = rt_data_freqs_surps[
        rt_data_freqs_surps['surprisal_s'].notna() &
        rt_data_freqs_surps['surprisal_p1_s'].notna() &
        rt_data_freqs_surps['surprisal_p2_s'].notna() &
        rt_data_freqs_surps['surprisal_p3_s'].notna() &
        rt_data_freqs_surps['logfreq_s'].notna() &
        rt_data_freqs_surps['logfreq_p1_s'].notna() &
        rt_data_freqs_surps['logfreq_p2_s'].notna() &
        rt_data_freqs_surps['logfreq_p3_s'].notna() &
        (rt_data_freqs_surps['sent_length'] != rt_data_freqs_surps['WordPosition'])
    ].copy()
    
    # Make predictions using the filler model
    try:
        # For statsmodels or similar models
        if hasattr(filler_model, 'predict'):
            rt_data_drop['predicted'] = filler_model.predict(rt_data_drop)
        else:
            # For other model types, might need different prediction method
            print(f"Warning: Could not make predictions for model {model}")
            rt_data_drop['predicted'] = np.nan
    except Exception as e:
        print(f"Error making predictions for model {model}: {e}")
        rt_data_drop['predicted'] = np.nan
    
    rt_data_drop['model'] = model
    
    return rt_data_drop