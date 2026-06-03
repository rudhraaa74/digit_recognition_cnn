"""
Phase 1: Environment Setup and Verification

This script verifies that all required libraries are installed and the data is accessible.

How to run it:
Open your terminal, ensure you are in the project root directory, and run:
python temp/verify.py
"""

# Import the necessary libraries
import torch           # PyTorch, the core deep learning library we will use
import torchvision     # Contains computer vision utilities (not heavily used here, but good to have)
import matplotlib      # For plotting and visualising our data and results later
import pandas as pd    # For loading and manipulating tabular data (our CSV files)
import numpy as np     # For numerical operations and array manipulations

def main():
    print("--- Environment Verification ---")
    
    # 1. Check PyTorch Version
    print(f"PyTorch version: {torch.__version__}")
    
    # 2. Check for CUDA (GPU)
    # CUDA allows PyTorch to run on NVIDIA GPUs for much faster training.
    # We print this to know what hardware is available to us.
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (GPU) available: {cuda_available}")
    
    if not cuda_available:
        print("Note: CUDA is not available. Training will run on CPU. This is fine for our dataset!")
        
    # 3. Check the Kaggle dataset accessibility
    # We attempt to read the first 5 rows to confirm the file is exactly where we expect it
    data_path = "data/digi_rec_train.csv"
    try:
        df = pd.read_csv(data_path, nrows=5)
        print("\n--- Data Verification ---")
        print(f"Successfully located {data_path}!")
        print(f"Shape of first 5 rows: {df.shape}")
        print("Expected shape: (5, 785) because there are 5 rows, 1 label column, and 784 pixel columns (28x28).")
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find {data_path}.")
        print("Please ensure the Kaggle dataset is properly extracted to the data/ folder.")

if __name__ == "__main__":
    main()
