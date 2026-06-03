import sys
import json
import nbformat

try:
    with open("temp/exp_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("Run run_exp45.py first to generate results!")
    sys.exit(1)
    
acc1_best = max(results["exp1"])
acc2_best = max(results["exp2"])
baseline = 99.15

notebook_path = "notebooks/experiments.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Exp 1 Markdown
md_exp1 = nbformat.v4.new_markdown_cell("""---
## Experiment 1: Dropout 0.25
**What we are changing:** We are keeping the exact same architecture as `src/model.py`, but we are lowering the Dropout rate from `0.5` to `0.25`.
**Why:** A dropout of 0.5 means 50% of neurons are zeroed out during training. This is very aggressive regularization. MNIST is a relatively simple dataset, so throwing away 50% of the network's capacity might be bottlenecking it. By dropping only 25%, the model retains more capacity to learn fine details, while still getting *some* regularization to prevent severe overfitting.""")

# 2. Exp 1 Code
code_exp1 = nbformat.v4.new_code_cell("""# Same architecture as src/model.py but with configurable dropout
class ExpDigitCNN(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(ExpDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout_rate) # Configurable dropout
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("--- Starting Experiment 1: Dropout 0.25 ---")
model_exp1 = ExpDigitCNN(dropout_rate=0.25).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_exp1.parameters(), lr=0.001)

# We will record the validation accuracy per epoch
val_accuracies_exp1 = []
best_acc_exp1 = 0.0

for epoch in range(1, 11):
    model_exp1.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model_exp1(images), labels)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model_exp1.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            _, predicted = torch.max(model_exp1(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    val_accuracies_exp1.append(val_acc)
    
    # Save the best model
    if val_acc > best_acc_exp1:
        best_acc_exp1 = val_acc
        torch.save(model_exp1.state_dict(), "../model/exp_dropout025.pth")
        
    print(f"Epoch {epoch}/10 | Val Acc: {val_acc:.2f}%")""")

# 3. Exp 2 Markdown
md_exp2 = nbformat.v4.new_markdown_cell("""---
## Experiment 2: ReduceLROnPlateau Scheduler + Dropout 0.25
**What we are changing:** We use the same Dropout 0.25 model from Experiment 1, but we introduce a `ReduceLROnPlateau` scheduler.
**Why:** The Adam optimizer uses a constant base learning rate (0.001). As the model gets close to the perfect weights, taking large "steps" (learning rate) can cause it to jump back and forth around the optimum (bouncing out of the minimum). `ReduceLROnPlateau` monitors our validation accuracy. If the accuracy stops improving for 2 epochs (`patience=2`), it cuts the learning rate in half (`factor=0.5`). This allows the model to take tiny, fine-tuned steps to settle into the absolute best weights!""")

# 4. Exp 2 Code
code_exp2 = nbformat.v4.new_code_cell("""from torch.optim.lr_scheduler import ReduceLROnPlateau

print("\\n--- Starting Experiment 2: Dropout 0.25 + Scheduler ---")
model_exp2 = ExpDigitCNN(dropout_rate=0.25).to(device)
optimizer2 = optim.Adam(model_exp2.parameters(), lr=0.001)

# The Scheduler: Monitors the metric (max accuracy). 
# If it doesn't improve for 2 epochs (patience), multiply LR by 0.5
scheduler = ReduceLROnPlateau(optimizer2, mode='max', patience=2, factor=0.5)

val_accuracies_exp2 = []
best_acc_exp2 = 0.0

for epoch in range(1, 11):
    # Print the current learning rate at the start of the epoch
    current_lr = optimizer2.param_groups[0]['lr']
    print(f"Epoch {epoch}/10 | Current LR: {current_lr}")
    
    model_exp2.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model_exp2(images), labels)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model_exp2.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            _, predicted = torch.max(model_exp2(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    val_accuracies_exp2.append(val_acc)
    
    # Step the scheduler (feed it the validation accuracy so it knows if we plateaued)
    scheduler.step(val_acc)
    
    if val_acc > best_acc_exp2:
        best_acc_exp2 = val_acc
        torch.save(model_exp2.state_dict(), "../model/exp_dropout025_scheduler.pth")
        
    print(f"   -> Val Acc: {val_acc:.2f}%")""")

# 5. Plotting Code
code_plot = nbformat.v4.new_code_cell(f"""import matplotlib.pyplot as plt

epochs = range(1, 11)

plt.figure(figsize=(10, 6))

# Plot Exp 1 & 2
plt.plot(epochs, val_accuracies_exp1, marker='o', label=f'Exp 1: Dropout 0.25 (Best: {{max(val_accuracies_exp1):.2f}}%)', color='blue')
plt.plot(epochs, val_accuracies_exp2, marker='s', label=f'Exp 2: Dropout 0.25 + LR Scheduler (Best: {{max(val_accuracies_exp2):.2f}}%)', color='orange')

# Plot Baseline
baseline_acc = 99.15
plt.axhline(baseline_acc, color='red', linestyle='--', label=f'Baseline: Dropout 0.5 (Best: {{baseline_acc}}%)')

plt.title('Validation Accuracy Across Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()""")

# 6. Final Comparison Markdown
final_md_text = f"""---
## Results Comparison & Conclusion

**Baseline (Dropout 0.5, No Scheduler):** 99.15%
**Experiment 1 (Dropout 0.25, No Scheduler):** {acc1_best:.2f}%
**Experiment 2 (Dropout 0.25 + Scheduler):** {acc2_best:.2f}%

### What do the numbers tell us?

1. **Why lowering dropout helped (or didn't):** 
   If Experiment 1 beat the baseline, it means Dropout 0.5 was too aggressive and suffocating the network. By lowering it to 0.25, the model had more functional neurons to recognize complex handwritten edges. If it performed slightly worse or the same, it means the model was already extracting all the useful features it could without overfitting.
   
2. **What the accuracy curves tell us about overfitting:** 
   If you look at the plot, a healthy model will rise steeply and then plateau. If a model is *overfitting*, the validation curve will hit a peak early on (e.g. Epoch 5) and then start degrading or jumping wildly as training continues. 
   
3. **The Power of the Scheduler:**
   The `ReduceLROnPlateau` scheduler watches the validation curve. If it flatlines for 2 epochs, it slices the learning rate in half. You can usually see this exact moment on the graph: the orange line (Exp 2) might plateau, and then suddenly shoot up slightly on the next epoch because the smaller learning rate allowed the optimizer to settle perfectly into the local minimum without overshooting!
"""
md_compare = nbformat.v4.new_markdown_cell(final_md_text)

# Append to notebook
nb.cells.extend([md_exp1, code_exp1, md_exp2, code_exp2, code_plot, md_compare])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
    
print("Notebook updated successfully!")
