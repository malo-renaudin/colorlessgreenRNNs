#!/usr/bin/env python3
"""
Interactive questionnaire for setting up CBR-RNN experiments.
"""
import subprocess
import sys
from datetime import datetime

def ask_question(question, default=None, options=None):
    """Ask a question and return the user's answer."""
    if options:
        print(f"\n{question}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        if default:
            prompt = f"Choose (1-{len(options)}) [default: {default}]: "
        else:
            prompt = f"Choose (1-{len(options)}): "
    else:
        if default:
            prompt = f"{question} [default: {default}]: "
        else:
            prompt = f"{question}: "
    
    response = input(prompt).strip()
    
    if not response and default:
        return default
    elif options:
        try:
            idx = int(response) - 1
            if 0 <= idx < len(options):
                return options[idx]
            else:
                print("Invalid choice. Please try again.")
                return ask_question(question, default, options)
        except ValueError:
            print("Please enter a number.")
            return ask_question(question, default, options)
    else:
        return response

def parse_list_input(user_input, data_type=int):
    """Parse space-separated list input."""
    try:
        return [data_type(x.strip()) for x in user_input.split() if x.strip()]
    except ValueError:
        print(f"Invalid input. Please enter space-separated {data_type.__name__} values.")
        return None

def main():
    print("="*60)
    print("🧠 CBR-RNN EXPERIMENT SETUP QUESTIONNAIRE")
    print("="*60)
    print("Welcome! I'll help you set up your experiment by asking a few questions.")
    print("You can press Enter to use default values shown in brackets.")
    
    # Experiment name
    exp_name = ask_question(
        "What would you like to name this experiment?",
        default=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Quick setup or custom
    setup_type = ask_question(
        "Choose setup type:",
        options=["Quick setup (recommended defaults)", "Custom setup (choose all parameters)"],
        default="Quick setup (recommended defaults)"
    )
    
    if setup_type == "Quick setup (recommended defaults)":
        # Predefined configurations
        config_choice = ask_question(
            "Choose a predefined configuration:",
            options=[
                "Small grid (fast) - 3 configs: dims=[256], heads=[4,8], temps=[1.0], gumbel=[true,false]",
                "Medium grid (balanced) - 18 configs: dims=[256,512], heads=[4,8,16], temps=[0.5,1.0,2.0], gumbel=[true,false]", 
                "Large grid (comprehensive) - 54 configs: dims=[128,256,512], heads=[4,8,16], temps=[0.1,0.5,1.0], gumbel=[true,false]",
                "Custom quick - choose just the key parameters"
            ]
        )
        
        if config_choice.startswith("Small grid"):
            hidden_dims = [256]
            num_heads = [4, 8]
            temperatures = [1.0]
            gumbel_softmax = ["true", "false"]
        elif config_choice.startswith("Medium grid"):
            hidden_dims = [256, 512]
            num_heads = [4, 8, 16]
            temperatures = [0.5, 1.0, 2.0]
            gumbel_softmax = ["true", "false"]
        elif config_choice.startswith("Large grid"):
            hidden_dims = [128, 256, 512]
            num_heads = [4, 8, 16]
            temperatures = [0.1, 0.5, 1.0]
            gumbel_softmax = ["true", "false"]
        else:  # Custom quick
            print("\nQuick custom setup - just choose the main parameters:")
            
            # Hidden dimensions
            while True:
                dims_input = ask_question(
                    "Hidden dimensions (space-separated)",
                    default="256 512"
                )
                hidden_dims = parse_list_input(dims_input, int)
                if hidden_dims:
                    break
            
            # Number of heads  
            while True:
                heads_input = ask_question(
                    "Number of attention heads (space-separated)",
                    default="4 8"
                )
                num_heads = parse_list_input(heads_input, int)
                if num_heads:
                    break
            
            # Temperatures
            while True:
                temps_input = ask_question(
                    "Temperatures (space-separated)",
                    default="0.5 1.0"
                )
                temperatures = parse_list_input(temps_input, float)
                if temperatures:
                    break
            
            # Gumbel softmax
            gumbel_choice = ask_question(
                "Test both with and without Gumbel softmax?",
                options=["Yes (test both)", "Only with Gumbel", "Only without Gumbel"],
                default="Yes (test both)"
            )
            
            if gumbel_choice == "Yes (test both)":
                gumbel_softmax = ["true", "false"]
            elif gumbel_choice == "Only with Gumbel":
                gumbel_softmax = ["true"]
            else:
                gumbel_softmax = ["false"]
    
    else:  # Custom setup
        print("\nCustom setup - specify all parameters:")
        
        # Hidden dimensions
        while True:
            dims_input = ask_question(
                "Hidden dimensions (space-separated integers)",
                default="128 256 512"
            )
            hidden_dims = parse_list_input(dims_input, int)
            if hidden_dims:
                break
        
        # Number of heads
        while True:
            heads_input = ask_question(
                "Number of attention heads (space-separated integers)",
                default="4 8 16"
            )
            num_heads = parse_list_input(heads_input, int)
            if num_heads:
                break
        
        # Temperatures
        while True:
            temps_input = ask_question(
                "Temperatures (space-separated floats)",
                default="0.1 0.5 1.0"
            )
            temperatures = parse_list_input(temps_input, float)
            if temperatures:
                break
        
        # Gumbel softmax
        gumbel_choice = ask_question(
            "Gumbel softmax options:",
            options=["Test both true and false", "Only true", "Only false"],
            default="Test both true and false"
        )
        
        if gumbel_choice == "Test both true and false":
            gumbel_softmax = ["true", "false"]
        elif gumbel_choice == "Only true":
            gumbel_softmax = ["true"]
        else:
            gumbel_softmax = ["false"]
    
    # Calculate total configurations
    total_configs = len(hidden_dims) * len(num_heads) * len(temperatures) * len(gumbel_softmax)
    
    # Show summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Experiment name: {exp_name}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Number of heads: {num_heads}")
    print(f"Temperatures: {temperatures}")
    print(f"Gumbel softmax: {gumbel_softmax}")
    print(f"Total configurations: {total_configs}")
    print("\nEach configuration will be:")
    print("✓ Trained for 40 epochs")
    print("✓ Evaluated on NounPP syntactic tasks")
    print("✓ Evaluated on BLiMP linguistic acceptability")
    print("✓ Evaluated on repeat surprisal working memory")
    print("="*60)
    
    # Confirm and launch
    confirm = ask_question(
        f"Ready to launch {total_configs} training+evaluation jobs?",
        options=["Yes, launch the experiment!", "No, let me modify something"],
        default="Yes, launch the experiment!"
    )
    
    if confirm == "No, let me modify something":
        print("Experiment cancelled. Run the script again to start over.")
        return
    
    print(f"\n🚀 Launching experiment '{exp_name}'...")
    bash_script = 'wm_xp/main.sh'
    import os
    if not os.path.exists(bash_script):
        print(f"\n❌ Could not find {bash_script}")
        print("Please update the bash_script variable in this questionnaire to match your script location.")
        print("Looking for these possible scripts:")
        possible_scripts = ["main.sh", "wm_xp/main.sh", "submit_experiment.sh", "simple_submit_script.sh"]
        for script in possible_scripts:
            if os.path.exists(script):
                print(f"  ✓ Found: {script}")
            else:
                print(f"  ✗ Not found: {script}")
        sys.exit(1)
    
    # Make it executable if it isn't
    import stat
    current_permissions = os.stat(bash_script).st_mode
    if not (current_permissions & stat.S_IXUSR):
        print(f"Making {bash_script} executable...")
        os.chmod(bash_script, current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    
    # Build command for bash script
    cmd = [
        bash_script,
        exp_name,
        " ".join(map(str, hidden_dims)),
        " ".join(map(str, num_heads)), 
        " ".join(map(str, temperatures)),
        " ".join(gumbel_softmax)
    ]
    
    print("Executing command:")
    print(" ".join([f'"{arg}"' if " " in str(arg) else str(arg) for arg in cmd]))
    print()
    
    try:
        # Launch the experiment via bash script
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Experiment '{exp_name}' launched successfully!")
        print("Monitor progress with: squeue -u $USER")
        print(f"Results will appear in: wm_xp/experiments/{exp_name}/results/")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching experiment: {e}")
        print("Please check your setup and try again.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ Could not find main.sh")
        print("Make sure the bash script is executable: chmod +x main.sh")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment setup cancelled by user.")
        sys.exit(0)