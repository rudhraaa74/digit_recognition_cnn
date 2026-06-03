"""
Phase 7 Experiment: What happens if we remove Dropout?
Goal: Observe overfitting in action.

Changes:
1. Removed the `nn.Dropout(0.5)` layer from the network.
2. The network will now memorize the training data much faster, but might perform slightly worse on the validation set because it hasn't learned to generalize!
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders

class NoDropoutCNN(nn.Module):
    def __init__(self):
        super(NoDropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # self.dropout = nn.Dropout(0.5) <--- REMOVED!
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x) <--- REMOVED!
        x = self.fc2(x)
        return x

def train_experiment():
    print("--- Starting Experiment: No Dropout (Overfitting Test) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    LEARNING_RATE = 0.001
    EPOCHS = 10
    
    train_loader, val_loader = get_dataloaders(csv_path="data/digi_rec_train.csv", batch_size=64)
    
    model = NoDropoutCNN().to(device)
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
    print("Notice how the Train Loss might be lower than baseline, but Val Acc might not be as good!")
    print(f"Baseline Val Acc: 98.95%")
    print(f"No-Dropout Val Acc: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train_experiment()
