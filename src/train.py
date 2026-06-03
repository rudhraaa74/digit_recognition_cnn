"""
Phase 4: Training Loop

This script handles the actual training process of our DigitCNN model.
It feeds data through the network, calculates the loss, adjusts the weights, 
and tracks the model's accuracy on the validation set.
"""

import os
import sys

# Add the parent directory to the path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import DigitCNN
from src.dataset import get_train_loader, get_val_loader

# --- Hyperparameters ---

# How big of a step the model takes when updating its weights. 
# Too high: it overshoots the optimal weights. Too low: it learns very slowly.
LEARNING_RATE = 0.001

# The number of times the model will see the entire training dataset.
EPOCHS = 10

# Where to save the best model weights so we can load them later for predictions.
MODEL_SAVE_PATH = "model/digit_cnn.pth"

def train():
    print("--- Starting Training Process ---")
    
    # 1. Setup Data
    # We load our training and validation datasets.
    train_loader = get_train_loader()
    val_loader = get_val_loader()
    
    # 2. Setup Device (CPU vs GPU)
    # MNIST trains fast enough on CPU, but if a GPU (CUDA) is available, we'll use it!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}\n")
    
    # 3. Setup Model, Loss Function, and Optimizer
    model = DigitCNN().to(device)
    
    # CrossEntropyLoss is the standard loss function for classification tasks.
    # It compares the model's predicted probabilities with the actual correct digit label.
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a smart optimizer. It automatically adjusts the learning rate for each individual weight
    # based on how much it has been changing, leading to faster and more stable training.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_accuracy = 0.0
    
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 4. The Training Loop
    for epoch in range(1, EPOCHS + 1):
        # -- TRAINING PHASE --
        model.train() # Tells PyTorch we are training (activates Dropout)
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Step A: Clear old gradients from the last step
            optimizer.zero_grad()
            
            # Step B: Forward Pass (make predictions)
            outputs = model(images)
            
            # Step C: Calculate the loss (how wrong the predictions were)
            loss = criterion(outputs, labels)
            
            # Step D: Backward Pass (calculate gradients - how to adjust each weight to reduce loss)
            loss.backward()
            
            # Step E: Optimizer Step (actually update the weights)
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # -- VALIDATION PHASE --
        # After seeing the training data, we check how well it performs on the validation data.
        model.eval() # Tells PyTorch we are evaluating (deactivates Dropout so we use all neurons)
        correct = 0
        total = 0
        
        # We don't need to calculate gradients during validation (saves memory and time)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                
                # The prediction is the index of the highest score
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        
        # Print progress for this epoch
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        
        # -- SAVE THE BEST MODEL --
        # We only save the model if it performs better on the validation set than before.
        # This guarantees we keep the best version, even if it overfits in later epochs.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Validation accuracy improved! Model saved to {MODEL_SAVE_PATH}")

    print("\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train()
