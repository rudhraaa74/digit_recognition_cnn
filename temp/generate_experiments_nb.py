import nbformat
import os

notebook_path = "notebooks/experiments.ipynb"

# Create a new notebook
nb = nbformat.v4.new_notebook()

# Intro
markdown_intro = nbformat.v4.new_markdown_cell("""# Phase 7: Experiments & Hyperparameter Tuning

In this notebook, we intentionally modify our baseline CNN to see what happens. This builds intuition on how hyperparameter tweaks affect model performance.

Our baseline validation accuracy (from Phase 4) was **98.95%**. Let's see if we can beat it!""")

# Setup imports
code_setup = nbformat.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
from src.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")""")

# Exp 1: Deeper Model
markdown_exp1 = nbformat.v4.new_markdown_cell("""---
## Experiment 1: The "Deeper is Better" Fallacy
We add a 3rd Convolutional Layer, add Batch Normalization (to speed up convergence), and increase the Dense layer to 256 neurons.

**Hypothesis:** A deeper model extracts more complex features, so it should perform better.
**Spoiler:** It usually slightly overfits on simple datasets like MNIST!""")

code_exp1_model = nbformat.v4.new_code_cell("""class DeeperDigitCNN(nn.Module):
    def __init__(self):
        super(DeeperDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x""")

code_exp1_train = nbformat.v4.new_code_cell("""print("Training Deeper Model...")
model_deeper = DeeperDigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_deeper.parameters(), lr=0.001)

train_loader, val_loader = get_dataloaders(csv_path="../data/digi_rec_train.csv", batch_size=64)

best_val_acc = 0.0
for epoch in range(1, 11):
    model_deeper.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_deeper(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Validation
    model_deeper.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            _, predicted = torch.max(model_deeper(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    if val_acc > best_val_acc: best_val_acc = val_acc
    print(f"Epoch {epoch}/10 | Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

print(f"\\nBest Validation Accuracy (Deeper Model): {best_val_acc:.2f}% (Baseline: 98.95%)")""")

# Exp 2: No Dropout
markdown_exp2 = nbformat.v4.new_markdown_cell("""---
## Experiment 2: Removing Dropout (Observing Overfitting)
Dropout randomly turns off 50% of the neurons during training to prevent the network from memorizing the data. Let's see what happens if we remove it!""")

code_exp2 = nbformat.v4.new_code_cell("""class NoDropoutCNN(nn.Module):
    def __init__(self):
        super(NoDropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

print("Training No-Dropout Model...")
model_nodropout = NoDropoutCNN().to(device)
optimizer = optim.Adam(model_nodropout.parameters(), lr=0.001)

best_val_acc_nodropout = 0.0
for epoch in range(1, 11):
    model_nodropout.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model_nodropout(images), labels)
        loss.backward()
        optimizer.step()
        
    model_nodropout.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            _, predicted = torch.max(model_nodropout(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    if val_acc > best_val_acc_nodropout: best_val_acc_nodropout = val_acc
    print(f"Epoch {epoch}/10 | Val Acc: {val_acc:.2f}%")""")

# Exp 3: Data Augmentation
markdown_exp3 = nbformat.v4.new_markdown_cell("""---
## Experiment 3: Data Augmentation (The True Accuracy Booster)
If we want to push past 99%, we need to show the model "new" data. We can do this artificially using Data Augmentation! 
During training, we will randomly rotate (±10 degrees) and translate (shift by 10%) the images slightly before feeding them to the model.

This forces the network to learn the *concept* of a digit rather than memorizing exact pixels! We will use the baseline architecture from Phase 3 but train it with augmented data for 15 epochs.""")

code_exp3 = nbformat.v4.new_code_cell("""from src.model import DigitCNN

print("Training Augmented Model...")
model_aug = DigitCNN().to(device)
optimizer = optim.Adam(model_aug.parameters(), lr=0.001)

# PyTorch transforms that operate on batches of Tensors
augment = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1))
])

best_val_acc_aug = 0.0
for epoch in range(1, 16): # More epochs since the task is harder now!
    model_aug.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Apply augmentation on the fly!
        images = augment(images)
        
        optimizer.zero_grad()
        loss = criterion(model_aug(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    model_aug.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            # Never augment validation data! We evaluate on pristine images.
            _, predicted = torch.max(model_aug(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    if val_acc > best_val_acc_aug: best_val_acc_aug = val_acc
    print(f"Epoch {epoch}/15 | Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

print(f"\\nBest Validation Accuracy (Augmented): {best_val_acc_aug:.2f}% (Baseline: 98.95%)")""")

nb.cells.extend([
    markdown_intro, code_setup,
    markdown_exp1, code_exp1_model, code_exp1_train,
    markdown_exp2, code_exp2,
    markdown_exp3, code_exp3
])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("experiments.ipynb created successfully!")
