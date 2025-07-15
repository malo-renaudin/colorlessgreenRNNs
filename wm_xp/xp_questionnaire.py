import argparse
import subprocess

def ask_question(prompt, default=None, type_func=str, options=None):
    """Ask a single question with validation"""
    while True:
        if options:
            prompt += f" ({'/'.join(map(str, options))})"
        if default is not None:
            prompt += f" [default: {default}]"
        
        answer = input(f"{prompt}: ").strip()
        
        if not answer and default is not None:
            return default
        
        if options and answer not in map(str, options):
            print(f"Please choose from: {options}")
            continue
            
        try:
            return type_func(answer)
        except ValueError:
            print(f"Invalid input. Expected {type_func.__name__}")

def ask_bool_list(prompt, default=None):
    """Ask for a list of boolean values"""
    if default:
        default_str = ' '.join(['true' if x else 'false' for x in default])
        prompt += f" [default: {default_str}]"
    
    answer = input(f"{prompt} (space-separated true/false): ").strip()
    
    if not answer and default:
        return default
        
    try:
        return [x.lower() == 'true' for x in answer.split()]
    except:
        print("Invalid input. Use 'true' or 'false' separated by spaces")
        return ask_bool_list(prompt, default)
def ask_list(prompt, default=None, type_func=int):
    """Ask for a space-separated list"""
    if default:
        prompt += f" [default: {' '.join(map(str, default))}]"
    
    answer = input(f"{prompt} (space-separated): ").strip()
    
    if not answer and default:
        return default
        
    try:
        return [type_func(x) for x in answer.split()]
    except ValueError:
        print(f"Invalid input. Expected list of {type_func.__name__}")
        return ask_list(prompt, default, type_func)

def experiment_questionnaire():
    """Interactive questionnaire for experiment setup"""
    print("=== Experiment Setup ===\n")
    
    # Basic info
    exp_name = ask_question("Experiment name")
    
    # Hyperparameters
    print("\n--- Hyperparameters ---")
    hidden_dims = ask_list("Hidden dimensions", [1024], int)
    num_heads = ask_list("Number of attention heads", [1], int)
    temperatures = ask_list("Temperature values", [0.1, 0.01, 0.001], float)
    
    gumbel_softmax = ask_bool_list("Gumbel softmax options", [True])
    
    # Scripts and settings
    print("\n--- Scripts ---")
    config_script = ask_question("Config script path", "wm_xp/grid_search/configs/generate_config_file.py")
    grid_script = ask_question("Grid search script path", "wm_xp/grid_search/gs.sh")
    
    print("\n--- Settings ---")
    output_dir = ask_question("Output directory", "wm_xp/experiments")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Experiment: {exp_name}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Num heads: {num_heads}")
    print(f"Temperatures: {temperatures}")
    print(f"Gumbel softmax: {gumbel_softmax}")
    print(f"Config script: {config_script}")
    print(f"Grid script: {grid_script}")
    print(f"Output: {output_dir}")
    
    confirm = ask_question("\nProceed with experiment", "y", options=["y", "n"])
    
    if confirm == "n":
        print("Cancelled.")
        return None
    
    # Run the main script with generated arguments
    print("\n=== Running Experiment ===")
    
    # Build command
    cmd = [
        "python", "wm_xp/main.py",  # Replace with your main script name
        "--name", exp_name,
        "--hidden_dims"] + [str(x) for x in hidden_dims] + [
        "--num_heads"] + [str(x) for x in num_heads] + [
        "--temperatures"] + [str(x) for x in temperatures] + [
        "--gumbel_softmax"] + ['true' if x else 'false' for x in gumbel_softmax] + [
        "--grid_search_script", grid_script,
        "--data", "/scratch2/mrenaudin/colorlessgreenRNNs/english_data",  # adjust as needed
        "--nounpp", "/scratch2/mrenaudin/colorlessgreenRNNs/NounPP/Stimuli/nounpp.txt",  # adjust as needed
        "--cuda"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Experiment completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment failed with exit code {e.returncode}")
        return False

if __name__ == "__main__":
    success = experiment_questionnaire()
    if success:
        print("Experiment setup and execution completed!")
    else:
        print("Experiment setup cancelled or failed.")