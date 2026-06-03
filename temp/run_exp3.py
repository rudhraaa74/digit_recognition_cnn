import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders

class ExpDigitCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ExpDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpDigitCNN(dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    train_loader, val_loader = get_dataloaders(csv_path="data/digi_rec_train.csv", batch_size=64)
    
    val_accuracies = []
    best_acc = 0.0
    
    print("\\n--- Starting Exp 3: Dropout 0.5 + Scheduler ---")
    
    for epoch in range(1, 11):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                _, predicted = torch.max(model(val_images).data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
                
        val_acc = 100.0 * correct / total
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/exp_dropout05_scheduler.pth")
            
        print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}%")
        
    # Append to existing results
    try:
        with open("temp/exp_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
        
    results["exp3"] = val_accuracies
    
    with open("temp/exp_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    train_experiment()
