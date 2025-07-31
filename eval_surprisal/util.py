import sys
import pandas as pd
import types

# Create the missing numeric module
numeric_module = types.ModuleType('pandas.core.indexes.numeric')

# Map old numeric index classes to their new locations
numeric_module.Int64Index = pd.Index
numeric_module.Float64Index = pd.Index  
numeric_module.UInt64Index = pd.Index
numeric_module.RangeIndex = pd.RangeIndex
numeric_module.Index = pd.Index

# Add some additional attributes that might be needed
from pandas.core.indexes import base
numeric_module.NumericIndex = base.Index

# Make sure all attributes are properly set
print("Setting up numeric module attributes:")
for attr_name in ['Int64Index', 'Float64Index', 'UInt64Index', 'RangeIndex', 'Index']:
    attr_value = getattr(numeric_module, attr_name)
    print(f"  {attr_name}: {attr_value}")

# Register the module
sys.modules['pandas.core.indexes.numeric'] = numeric_module

# Verify the module is accessible
test_module = sys.modules.get('pandas.core.indexes.numeric')
print(f"Module registered: {test_module}")
print(f"Module attributes: {dir(test_module)}")
print(f"Int64Index accessible: {hasattr(test_module, 'Int64Index')}")
import pickle
import numpy as np 
import sys
import joblib
import sys
import pandas as pd

# sys.modules['pandas.core.indexes.numeric'] = numeric

# # Print current pandas version and available modules
# print(f"Pandas version: {pd.__version__}")
# print("Available pandas.core.indexes modules:")
# import pandas.core.indexes
# print(dir(pandas.core.indexes))

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
        # Define the feature columns that the filler model expects
        # These should match what the model was trained on
        feature_columns = [
            'surprisal_s', 'surprisal_p1_s', 'surprisal_p2_s', 'surprisal_p3_s',
            'logfreq_s', 'logfreq_p1_s', 'logfreq_p2_s', 'logfreq_p3_s',
            'length_s', 'length_p1_s', 'length_p2_s', 'length_p3_s'
        ]
        
        # Check which features are available
        available_features = [col for col in feature_columns if col in rt_data_drop.columns]
        missing_features = [col for col in feature_columns if col not in rt_data_drop.columns]
        
        if missing_features:
            print(f"Warning: Missing features for model {model}: {missing_features}")
        
        print(f"Using features for prediction: {available_features}")
        
        # Extract feature matrix for prediction
        X_pred = rt_data_drop[available_features]
        
        # Check for any remaining NaN values in features
        if X_pred.isnull().any().any():
            print(f"Warning: NaN values found in features for model {model}")
            print("NaN counts per feature:")
            print(X_pred.isnull().sum())
        
        # Make predictions
        if hasattr(filler_model, 'predict'):
            rt_data_drop['predicted'] = filler_model.predict(X_pred)
            print(f"Successfully made {len(rt_data_drop)} predictions for model {model}")
        else:
            print(f"Warning: Model {model} does not have predict method")
            rt_data_drop['predicted'] = np.nan
            
    except Exception as e:
        print(f"Error making predictions for model {model}: {e}")
        print(f"Model type: {type(filler_model)}")
        if hasattr(filler_model, 'feature_names_in_'):
            print(f"Model expects features: {filler_model.feature_names_in_}")
        rt_data_drop['predicted'] = np.nan
    
    rt_data_drop['model'] = model
    return rt_data_drop