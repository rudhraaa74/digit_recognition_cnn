"""
Phase 6: Deep Dive into Test Predictions

This script analyzes the model's confidence on the 28,000 test images.
"""

import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import DigitCNN

MODEL_PATH = "model/digit_cnn.pth"
TEST_CSV_PATH = "data/digi_rec_test.csv"
REPORT_PATH = "model/confidence_report.csv"

def analyse():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Confidence Analysis on {device} ---")
    
    # 1. Load Model
    model = DigitCNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    # 2. Load Data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_pixels = test_df.values
    
    print(f"Loaded {len(test_pixels)} images. Running batch inference...")
    
    # Preprocess all data efficiently
    test_tensor = torch.tensor(test_pixels, dtype=torch.float32) / 255.0
    test_tensor = (test_tensor - 0.1307) / 0.3081
    test_tensor = test_tensor.reshape(-1, 1, 28, 28)
    
    all_confidences = []
    all_preds = []
    
    # Process in batches to avoid running out of memory
    batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            
            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Get max probability (confidence) and the predicted class
            conf, pred = torch.max(probs, dim=1)
            
            all_confidences.extend(conf.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            
    print("Inference complete. Finding top and bottom predictions...")
    
    # 3. Create a DataFrame for analysis
    # ImageId is 1-indexed for Kaggle
    results = pd.DataFrame({
        "ImageId": range(1, len(all_preds) + 1),
        "PredictedLabel": all_preds,
        "Confidence": all_confidences
    })
    
    # 4. Find Top 10 most confident per digit (closest to 1.0)
    print("\n--- Top 10 Most Confident Predictions per Digit ---")
    for digit in range(10):
        digit_df = results[results["PredictedLabel"] == digit]
        top_10 = digit_df.nlargest(10, "Confidence")
        avg_conf = digit_df["Confidence"].mean() * 100
        print(f"Digit {digit}: Average Confidence {avg_conf:.2f}%. Top confidence: {top_10['Confidence'].max()*100:.4f}%")
        
    # 5. Find Top 20 least confident predictions overall
    print("\n--- Top 20 Least Confident Predictions Overall ---")
    bottom_20 = results.nsmallest(20, "Confidence")
    print(bottom_20.to_string(index=False))
    
    # 6. Save Report
    results.to_csv(REPORT_PATH, index=False)
    print(f"\nSaved full confidence report to {REPORT_PATH}")

if __name__ == "__main__":
    analyse()
