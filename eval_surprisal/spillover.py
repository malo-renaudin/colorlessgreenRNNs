import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import pickle
import gc
from datetime import datetime

class SpilloverAnalysis:
    """
    Python implementation of spillover analysis for psycholinguistic data.
    Fits linear mixed effects models mapping surprisals 0-3 words back to reading times.
    """
    
    def __init__(self):
        self.models = {}
        
    def load_data(self, subset_name="Fillers"):
        """Load SPR data - placeholder for actual data loading function"""
        # This would need to be implemented based on your specific data format
        print(f"Loading SPR data for {subset_name}")
        # Return mock structure for now
        return pd.DataFrame()
    
    def load_surprisal_data(self):
        """Load surprisal data from different models"""
        print("Loading surprisal data...")
        
        # Load surprisal data
        surps_lstm = pd.read_csv("../data/lstm/items_filler.lstm.csv.scaled")
        surps_gpt2 = pd.read_csv("../data/gpt2/items_filler.gpt2.csv.scaled") 
        surps_rnng = pd.read_csv("../data/rnng/items_filler.rnng.csv.scaled")
        
        # Adjust to 1-indexing (R-style)
        surps_lstm['word_pos'] = surps_lstm['word_pos'] + 1
        surps_gpt2['word_pos'] = surps_gpt2['word_pos'] + 1
        surps_rnng['word_pos'] = surps_rnng['word_pos'] + 1
        
        return {
            'lstm': surps_lstm,
            'gpt2': surps_gpt2, 
            'rnng': surps_rnng
        }
    
    def bind_surps(self, spr, surps):
        """
        Merge SPR data with surprisal data and create lagged variables
        """
        # Merge datasets
        merged = pd.merge(spr, surps, 
                         left_on=['Sentence', 'WordPosition'],
                         right_on=['Sentence', 'word_pos'],
                         how='left')
        
        # Clean up columns
        if 'item_x' in merged.columns:
            merged['item'] = merged['item_x']
        merged['surprisal_s'] = merged['sum_surprisal_s']  # change to mean_surprisal_s if needed
        
        # Create lagged variables grouped by item and participant
        def create_lags(group):
            """Create lag variables for a group (item + participant)"""
            # Sort by word position to ensure correct order
            group = group.sort_values('WordPosition')
            
            # Reading time lags
            group['RT_p1'] = group['RT'].shift(1)
            group['RT_p2'] = group['RT'].shift(2) 
            group['RT_p3'] = group['RT'].shift(3)
            
            # Length lags
            group['length_p1_s'] = group['length_s'].shift(1)
            group['length_p2_s'] = group['length_s'].shift(2)
            group['length_p3_s'] = group['length_s'].shift(3)
            
            # Log frequency lags
            group['logfreq_p1_s'] = group['logfreq_s'].shift(1)
            group['logfreq_p2_s'] = group['logfreq_s'].shift(2)
            group['logfreq_p3_s'] = group['logfreq_s'].shift(3)
            
            # Surprisal lags
            group['surprisal_p1_s'] = group['surprisal_s'].shift(1)
            group['surprisal_p2_s'] = group['surprisal_s'].shift(2)
            group['surprisal_p3_s'] = group['surprisal_s'].shift(3)
            
            return group
        
        # Apply lag creation to each group
        with_lags = merged.groupby(['item', 'participant']).apply(create_lags).reset_index(drop=True)
        
        # Calculate sentence length
        with_lags['sent_length'] = with_lags['Sentence'].str.split().str.len()
        
        # Filter out rows with missing data or sentence-final words
        required_cols = [
            'surprisal_s', 'surprisal_p1_s', 'surprisal_p2_s', 'surprisal_p3_s',
            'logfreq_s', 'logfreq_p1_s', 'logfreq_p2_s', 'logfreq_p3_s'
        ]
        
        # Create mask for complete cases
        complete_mask = True
        for col in required_cols:
            complete_mask = complete_mask & with_lags[col].notna()
        
        # Exclude sentence-final words
        not_final_mask = with_lags['sent_length'] != with_lags['WordPosition']
        
        dropped = with_lags[complete_mask & not_final_mask].copy()
        
        print(f"Dropped: {len(with_lags) - len(dropped)} rows")
        
        return dropped
    
    def standardize_column(self, df, col_name):
        """Standardize a column (z-score)"""
        return (df[col_name] - df[col_name].mean()) / df[col_name].std()
    
    def fit_filler_model(self, data, model_name):
        """
        Fit linear mixed effects model for filler items
        """
        print(f"Fitting filler model for {model_name}...")
        
        # Standardize WordPosition
        data = data.copy()
        data['WordPosition_s'] = self.standardize_column(data, 'WordPosition')
        
        # Define the formula
        # Note: statsmodels uses a different syntax than R's lme4
        formula = """RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s + 
                     WordPosition_s + logfreq_s + length_s + logfreq_p1_s + length_p1_s + 
                     logfreq_p2_s + length_p2_s + logfreq_p3_s + length_p3_s +
                     logfreq_s:length_s + logfreq_p1_s:length_p1_s + 
                     logfreq_p2_s:length_p2_s + logfreq_p3_s:length_p3_s"""
        
        try:
            # Fit mixed effects model
            # Note: statsmodels MixedLM has different syntax than R's lmer
            model = MixedLM.from_formula(
                formula, 
                data, 
                groups=data["participant"],
                re_formula="surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s"
            )
            
            result = model.fit(method=['lbfgs'])
            
            print(f"Model fitted successfully for {model_name}")
            print(result.summary())
            
            return result
            
        except Exception as e:
            print(f"Error fitting model for {model_name}: {e}")
            
            # Fallback to simpler model if convergence fails
            simple_formula = "RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s + WordPosition_s"
            
            model = MixedLM.from_formula(
                simple_formula,
                data,
                groups=data["participant"] 
            )
            
            result = model.fit()
            print(f"Fitted simplified model for {model_name}")
            
            return result
    
    def run_analysis(self):
        """
        Main analysis pipeline
        """
        print(f"Starting Spillover Analysis - {datetime.now()}")
        
        # Load data
        spr = self.load_data("Fillers")
        surprisal_data = self.load_surprisal_data()
        
        # Process each model type
        for model_name, surps in surprisal_data.items():
            print(f"\n{'='*50}")
            print(f"Processing {model_name.upper()} model")
            print(f"{'='*50}")
            
            # Bind surprisal data with SPR data
            data = self.bind_surps(spr, surps)
            
            if len(data) == 0:
                print(f"No data available for {model_name}, skipping...")
                continue
                
            # Fit mixed effects model
            fitted_model = self.fit_filler_model(data, model_name)
            
            # Store model
            self.models[model_name] = fitted_model
            
            # Save model to file
            with open(f"filler_models/filler_{model_name}_sum.pkl", 'wb') as f:
                pickle.dump(fitted_model, f)
            
            print(f"Model saved: filler_models/filler_{model_name}_sum.pkl")
            
            # Free memory
            del data
            gc.collect()
            
        print(f"\nAnalysis completed - {datetime.now()}")
        
    def load_saved_model(self, model_name):
        """Load a previously saved model"""
        with open(f"filler_models/filler_{model_name}_sum.pkl", 'rb') as f:
            return pickle.load(f)
    
    def predict_reading_times(self, model_name, new_data):
        """Use fitted model to predict reading times for new data"""
        if model_name not in self.models:
            self.models[model_name] = self.load_saved_model(model_name)
            
        model = self.models[model_name]
        predictions = model.predict(new_data)
        
        return predictions

# Usage example
if __name__ == "__main__":
    # Create analysis object
    analysis = SpilloverAnalysis()
    
    # Run the full analysis pipeline
    analysis.run_analysis()
    
    # Example of loading and using a saved model
    # predictions = analysis.predict_reading_times('lstm', new_surprisal_data)