#!/usr/bin/env python3
"""
Utility script to set dataset percentage for training
Usage: python set_dataset_percentage.py [percentage]
Example: python set_dataset_percentage.py 75
"""

import json
import sys
import os

def update_parameter_file(percentage, parameter_file="parameter_enhanced.json"):
    """
    Update the dataset_percentage in a parameter file
    
    Args:
        percentage: Dataset percentage (1-100)
        parameter_file: Path to parameter file
    """
    if not os.path.exists(parameter_file):
        print(f"Error: Parameter file {parameter_file} not found!")
        return False
    
    # Load current parameters
    with open(parameter_file, 'r') as f:
        params = json.load(f)
    
    # Update dataset percentage
    params['dataset_percentage'] = float(percentage)
    
    # Save updated parameters
    with open(parameter_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"✅ Updated {parameter_file} with dataset_percentage = {percentage}%")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python set_dataset_percentage.py [percentage]")
        print("Example: python set_dataset_percentage.py 75")
        print("\nAvailable parameter files:")
        
        # List available parameter files
        param_files = [f for f in os.listdir('.') if f.startswith('parameter') and f.endswith('.json')]
        for f in param_files:
            print(f"  - {f}")
        
        return
    
    try:
        percentage = float(sys.argv[1])
        if percentage < 1 or percentage > 100:
            print("Error: Percentage must be between 1 and 100")
            return
    except ValueError:
        print("Error: Percentage must be a valid number")
        return
    
    # Update all parameter files
    param_files = [f for f in os.listdir('.') if f.startswith('parameter') and f.endswith('.json')]
    
    if not param_files:
        print("No parameter files found!")
        return
    
    print(f"Setting dataset percentage to {percentage}% in all parameter files...")
    
    for param_file in param_files:
        update_parameter_file(percentage, param_file)
    
    print(f"\n🎯 All parameter files updated! You can now train with {percentage}% of your dataset.")
    print(f"💡 This will reduce training time and memory usage significantly.")

if __name__ == "__main__":
    main()

