"""
Phase 7 Experiment: Improved Deeper Architecture
Goal: Try to beat our 98.95% validation baseline!

Changes:
1. Added a 3rd Convolutional Layer to extract even more complex patterns.
2. Added Batch Normalization after each Conv layer to stabilize and speed up training.
3. Tweaked the Dense layer to 256 neurons.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders

class DeeperDigitCNN(nn.Module):
    def __init__(self):
        super(DeeperDigitCNN, self).__init__()
        # Layer 1: 1 input channel -> 32 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: 32 -> 64 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: 64 -> 128 filters
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # After 3 MaxPools (each halving the 28x28 image), the spatial size will be:
        # 28 -> 14 -> 7 -> 3. (Actually 28 -> 14, 14 -> 7. 7/2 = 3).
        # Wait, maxpool of 7x7 with kernel 2 is 3x3.
        # So 128 channels * 3 * 3 = 1152 features.
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten and Dense
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_experiment():
    print("--- Starting Experiment: Deeper CNN with Batch Norm ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    LEARNING_RATE = 0.001
    EPOCHS = 10
    BATCH_SIZE = 64
    
    train_loader, val_loader = get_dataloaders(csv_path="data/digi_rec_train.csv", batch_size=BATCH_SIZE)
    
    model = DeeperDigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_accuracy = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
                
        val_acc = 100.0 * correct / total
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            
    print("\n--- Training Complete ---")
    print(f"Baseline Val Acc: 98.95%")
    print(f"New Best Val Acc: {best_val_accuracy:.2f}%")
    print(f"Improvement: {best_val_accuracy - 98.95:.2f}%")

if __name__ == "__main__":
    train_experiment()
