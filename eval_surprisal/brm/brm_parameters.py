import bambi as bmb

def get_brms_parameters(prior_type):
    """
    Get BRMS parameters adapted for bambi/PyMC
    
    Parameters:
    -----------
    prior_type : str
        Type of prior ('prior1', 'prior2', 'prior3', 'prior_bernoulli')
    
    Returns:
    --------
    dict
        Dictionary containing priors and sampling parameters
    """
    
    if prior_type == 'prior1':
        curr_prior = {
            "Intercept": bmb.Prior("Normal", mu=300, sigma=1000),
            "b": bmb.Prior("Normal", mu=0, sigma=150),        # coefficients
            "sd": bmb.Prior("Normal", mu=0, sigma=200),       # random effects std
            "sigma": bmb.Prior("Normal", mu=0, sigma=500)     # residual std
        }
    elif prior_type == 'prior2':
        curr_prior = {
            "Intercept": bmb.Prior("Normal", mu=300, sigma=1000),
            "b": bmb.Prior("Normal", mu=0, sigma=100),        # more informative
            "sd": bmb.Prior("Normal", mu=0, sigma=200),
            "sigma": bmb.Prior("Normal", mu=0, sigma=500)
        }
    elif prior_type == 'prior3':
        curr_prior = {
            "Intercept": bmb.Prior("Normal", mu=300, sigma=1000),
            "b": bmb.Prior("Normal", mu=0, sigma=100),        # most informative
            "sd": bmb.Prior("Normal", mu=0, sigma=150),       # tighter random effects
            "sigma": bmb.Prior("Normal", mu=0, sigma=300)     # tighter residuals
        }
    elif prior_type == 'prior_bernoulli':
        curr_prior = {
            "Intercept": bmb.Prior("Normal", mu=-1.5, sigma=1),
            "b": bmb.Prior("Normal", mu=0, sigma=0.75)
        }
    else:
        raise ValueError('ENTER A VALID PRIOR TYPE')
    
    params = {
        'priors': curr_prior,
        'ncores': 4,
        'niters': 12000,
        'seed': 117,
        'warmup': 6000,
        'adapt_delta': 0.8
    }
    
    return params

def get_bambi_priors_dict(prior_type, formula_terms=None):
    """
    Get priors formatted specifically for bambi model fitting
    
    Parameters:
    -----------
    prior_type : str
        Type of prior
    formula_terms : list, optional
        Specific terms in the model formula to set priors for
    
    Returns:
    --------
    dict
        Priors dictionary formatted for bambi
    """
    base_params = get_brms_parameters(prior_type)
    priors = base_params['priors']
    
    # If specific formula terms are provided, map them appropriately
    if formula_terms:
        bambi_priors = {}
        for term in formula_terms:
            if term == 'Intercept':
                bambi_priors["Intercept"] = priors["Intercept"]
            elif 'pGram_coded' in term or any(coef in term for coef in ['pGram', 'coded']):
                bambi_priors[term] = priors["b"]
            elif '|' in term:  # Random effects
                bambi_priors[term] = priors["sd"]
            else:
                bambi_priors[term] = priors["b"]  # Default to coefficient prior
        
        # Always include sigma for residual variance
        bambi_priors["sigma"] = priors["sigma"]
        return bambi_priors
    
    return priors

def fit_bambi_model_with_priors(data, formula, prior_type, model_name=None, roi=None):
    """
    Fit a bambi model with specified priors and parameters
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data for model fitting
    formula : str
        Model formula
    prior_type : str
        Prior type ('prior1', 'prior2', 'prior3')
    model_name : str, optional
        Model name for filtering data
    roi : int, optional
        ROI for filtering data
    
    Returns:
    --------
    bambi fitted model
    """
    # Get parameters
    params = get_brms_parameters(prior_type)
    
    # Filter data if needed
    if model_name and roi is not None:
        filtered_data = data[
            (data['Type'] == "AGREE") & 
            (data['ROI'] == roi) & 
            (data['model'] == model_name) & 
            (data['RT'].notna())
        ].copy()
    else:
        filtered_data = data.copy()
    
    # Create model
    model = bmb.Model(formula, filtered_data)
    
    # Extract formula terms for prior mapping
    # This is a simplified approach - you might need to adjust based on your specific formula
    formula_terms = ['Intercept', 'pGram_coded', '1|item', '1|participant', 
                    'pGram_coded|item', 'pGram_coded|participant']
    
    # Get priors
    priors = get_bambi_priors_dict(prior_type, formula_terms)
    
    # Fit model
    fitted_model = model.fit(
        draws=params['niters'] - params['warmup'],
        tune=params['warmup'],
        chains=params['ncores'],
        random_seed=params['seed'],
        target_accept=params['adapt_delta'],
        priors=priors
    )
    
    return fitted_model

# Example usage function for your specific case
def fit_agreement_model(data, model_name, roi, prior_type='prior1'):
    """
    Fit agreement model with specified parameters
    """
    formula = "predicted ~ pGram_coded + (1|item) + (1|participant) + (pGram_coded|item) + (pGram_coded|participant)"
    
    return fit_bambi_model_with_priors(
        data=data,
        formula=formula, 
        prior_type=prior_type,
        model_name=model_name,
        roi=roi
    )