"""
Phase 5: Evaluation on Test Data

This script measures real performance on the Kaggle test set and the validation set.
"""

import os
import sys

# Add parent directory to sys.path so we can run this from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
import torch.nn as nn
from src.model import DigitCNN
from src.dataset import get_val_loader

MODEL_PATH = "model/digit_cnn.pth"
TEST_CSV_PATH = "data/digi_rec_test.csv"
PREDICTIONS_PATH = "model/predictions.csv"

def evaluate_validation(model, device):
    print("--- Evaluating on Validation Set ---")
    val_loader = get_val_loader(csv_path="data/digi_rec_train.csv")
    
    correct = 0
    total = 0
    
    # We don't need gradients for evaluation
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%\n")
    return val_acc

def generate_test_predictions(model, device):
    print("--- Generating Predictions on Kaggle Test Set ---")
    
    # 1. Load the Kaggle test CSV (no labels, only pixels)
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_pixels = test_df.values
    
    predictions = []
    
    print(f"Loaded {len(test_pixels)} test images. Running inference...")
    
    # We don't need gradients for inference
    with torch.no_grad():
        for i in range(len(test_pixels)):
            # 2. Preprocess exactly the same as training
            pixel_row = test_pixels[i] / 255.0
            pixel_row = (pixel_row - 0.1307) / 0.3081
            
            # 3. Reshape and convert to Tensor
            image_tensor = torch.tensor(pixel_row, dtype=torch.float32).reshape(1, 1, 28, 28).to(device)
            
            # 4. Predict
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            
            # 5. Save prediction
            # ImageId is 1-indexed for Kaggle submissions
            predictions.append({"ImageId": i + 1, "Label": predicted.item()})
            
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1} images...")
                
    # 6. Save to CSV
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved {len(predictions)} predictions to {PREDICTIONS_PATH}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}\n")
    
    model = DigitCNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Set model to evaluation mode (disables dropout)
        print("Loaded model weights successfully.\n")
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Make sure you've trained the model first.")
        return

    # Run evaluations
    evaluate_validation(model, device)
    generate_test_predictions(model, device)

if __name__ == "__main__":
    main()
