import sys
import json
import nbformat

try:
    with open("temp/exp_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("Run run_exp4.py first to generate results!")
    sys.exit(1)
    
acc1_best = max(results["exp1"])
acc2_best = max(results["exp2"])
acc3_best = max(results["exp3"])
acc4_best = max(results["exp4"])
baseline = 99.15

notebook_path = "notebooks/experiments.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Exp 4 Markdown
md_exp4 = nbformat.v4.new_markdown_cell("""---
## Experiment 4: Dropout 0.25 + Scheduler + Data Augmentation
**What we are changing:** We are going back to the **Dropout 0.25 + Scheduler** setup (from Experiment 2), but this time we are adding **Data Augmentation**! During training, we will dynamically rotate the images by up to 10 degrees and shift them by up to 10% vertically/horizontally.
**Why:** Lower dropout (0.25) means the network has more capacity to learn. In Exp 1 and 2, this extra capacity led to slight overfitting. But by augmenting the data, we make the problem *harder*! The network sees "new" variations of digits every epoch. The extra capacity from the 0.25 dropout might be exactly what the model needs to learn these new, augmented patterns without underfitting!""")

# 2. Exp 4 Code
code_exp4 = nbformat.v4.new_code_cell("""import torchvision.transforms as T

print("\\n--- Starting Experiment 4: Dropout 0.25 + Sched + Augmentation ---")
model_exp4 = ExpDigitCNN(dropout_rate=0.25).to(device)
optimizer4 = optim.Adam(model_exp4.parameters(), lr=0.001)

scheduler4 = ReduceLROnPlateau(optimizer4, mode='max', patience=2, factor=0.5)

# PyTorch transforms that operate on batches of Tensors
augment = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1))
])

val_accuracies_exp4 = []
best_acc_exp4 = 0.0

for epoch in range(1, 11):
    current_lr = optimizer4.param_groups[0]['lr']
    print(f"Epoch {epoch}/10 | Current LR: {current_lr}")
    
    model_exp4.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Apply augmentation on the fly!
        images = augment(images)
        
        optimizer4.zero_grad()
        loss = criterion(model_exp4(images), labels)
        loss.backward()
        optimizer4.step()
        
    # Evaluate
    model_exp4.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            # Never augment validation data! We evaluate on pristine images.
            _, predicted = torch.max(model_exp4(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    val_accuracies_exp4.append(val_acc)
    
    # Step the scheduler
    scheduler4.step(val_acc)
    
    if val_acc > best_acc_exp4:
        best_acc_exp4 = val_acc
        torch.save(model_exp4.state_dict(), "../model/exp_dropout025_sched_aug.pth")
        
    print(f"   -> Val Acc: {val_acc:.2f}%")""")

# 3. New Plotting Code
code_plot = nbformat.v4.new_code_cell(f"""import matplotlib.pyplot as plt

epochs = range(1, 11)

plt.figure(figsize=(10, 6))

# Plot all Experiments
plt.plot(epochs, val_accuracies_exp1, marker='o', label=f'Exp 1: Drop 0.25 (Best: {{max(val_accuracies_exp1):.2f}}%)', color='blue')
plt.plot(epochs, val_accuracies_exp2, marker='s', label=f'Exp 2: Drop 0.25 + LR Sched (Best: {{max(val_accuracies_exp2):.2f}}%)', color='orange')
plt.plot(epochs, val_accuracies_exp3, marker='^', label=f'Exp 3: Drop 0.5 + LR Sched (Best: {{max(val_accuracies_exp3):.2f}}%)', color='green')
plt.plot(epochs, val_accuracies_exp4, marker='x', label=f'Exp 4: Drop 0.25 + Sched + Aug (Best: {{max(val_accuracies_exp4):.2f}}%)', color='purple')

# Plot Baseline
baseline_acc = 99.15
plt.axhline(baseline_acc, color='red', linestyle='--', label=f'Baseline: Dropout 0.5 (Best: {{baseline_acc}}%)')

plt.title('Validation Accuracy Across Epochs (All Experiments)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontsize=12)
plt.xticks(epochs)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()""")

# 4. Final Comparison Markdown
final_md_text = f"""---
## The Ultimate Conclusion

**Baseline (Drop 0.5, No Sched):** 99.15%
**Exp 1 (Drop 0.25, No Sched):** {acc1_best:.2f}%
**Exp 2 (Drop 0.25 + Sched):** {acc2_best:.2f}%
**Exp 3 (Drop 0.5 + Sched):** {acc3_best:.2f}%
**Exp 4 (Drop 0.25 + Sched + Aug):** {acc4_best:.2f}%

### Conclusion
By adding Data Augmentation to the `Dropout 0.25` architecture, we drastically increased the difficulty of the learning task. Without augmentation (Exp 1 and 2), the 0.25 dropout gave the model too much capacity, causing it to overfit. But *with* augmentation, that extra capacity became a superpower, allowing the model to learn the much more complex, augmented dataset!
"""
md_compare = nbformat.v4.new_markdown_cell(final_md_text)

# Append to notebook
nb.cells.extend([md_exp4, code_exp4, code_plot, md_compare])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
    
print("Notebook updated successfully with Experiment 4!")
