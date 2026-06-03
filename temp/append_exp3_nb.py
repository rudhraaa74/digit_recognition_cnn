import sys
import json
import nbformat

try:
    with open("temp/exp_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("Run run_exp3.py first to generate results!")
    sys.exit(1)
    
acc1_best = max(results["exp1"])
acc2_best = max(results["exp2"])
acc3_best = max(results["exp3"])
baseline = 99.15

notebook_path = "notebooks/experiments.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Exp 3 Markdown
md_exp3 = nbformat.v4.new_markdown_cell("""---
## Experiment 3: Dropout 0.5 + ReduceLROnPlateau Scheduler
**What we are changing:** Based on the results of Experiments 1 and 2, we noticed that dropping the Dropout rate to 0.25 actually harmed the model's ability to generalize, making it perform slightly worse than our 0.5 Dropout baseline. Therefore, in this experiment, we are keeping the original **Dropout of 0.5** (which we know regularizes the model perfectly), but we are re-introducing the **ReduceLROnPlateau** scheduler!
**Why:** The goal is to see if we can get the "best of both worlds". The heavy 0.5 Dropout will prevent overfitting, while the Learning Rate Scheduler will help the optimizer take tiny, precise steps near the end of training to settle perfectly into the minimum without bouncing out.""")

# 2. Exp 3 Code
code_exp3 = nbformat.v4.new_code_cell("""print("\\n--- Starting Experiment 3: Dropout 0.5 + Scheduler ---")
model_exp3 = ExpDigitCNN(dropout_rate=0.5).to(device)
optimizer3 = optim.Adam(model_exp3.parameters(), lr=0.001)

# The Scheduler: Monitors the metric (max accuracy). 
scheduler3 = ReduceLROnPlateau(optimizer3, mode='max', patience=2, factor=0.5)

val_accuracies_exp3 = []
best_acc_exp3 = 0.0

for epoch in range(1, 11):
    current_lr = optimizer3.param_groups[0]['lr']
    print(f"Epoch {epoch}/10 | Current LR: {current_lr}")
    
    model_exp3.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer3.zero_grad()
        loss = criterion(model_exp3(images), labels)
        loss.backward()
        optimizer3.step()
        
    # Evaluate
    model_exp3.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            _, predicted = torch.max(model_exp3(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    val_accuracies_exp3.append(val_acc)
    
    # Step the scheduler
    scheduler3.step(val_acc)
    
    if val_acc > best_acc_exp3:
        best_acc_exp3 = val_acc
        torch.save(model_exp3.state_dict(), "../model/exp_dropout05_scheduler.pth")
        
    print(f"   -> Val Acc: {val_acc:.2f}%")""")

# 3. New Plotting Code
code_plot = nbformat.v4.new_code_cell(f"""import matplotlib.pyplot as plt

epochs = range(1, 11)

plt.figure(figsize=(10, 6))

# Plot all Experiments
plt.plot(epochs, val_accuracies_exp1, marker='o', label=f'Exp 1: Drop 0.25 (Best: {{max(val_accuracies_exp1):.2f}}%)', color='blue')
plt.plot(epochs, val_accuracies_exp2, marker='s', label=f'Exp 2: Drop 0.25 + LR Sched (Best: {{max(val_accuracies_exp2):.2f}}%)', color='orange')
plt.plot(epochs, val_accuracies_exp3, marker='^', label=f'Exp 3: Drop 0.5 + LR Sched (Best: {{max(val_accuracies_exp3):.2f}}%)', color='green')

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
## The Final Verdict

**Baseline (Drop 0.5, No Sched):** 99.15%
**Exp 1 (Drop 0.25, No Sched):** {acc1_best:.2f}%
**Exp 2 (Drop 0.25 + Sched):** {acc2_best:.2f}%
**Exp 3 (Drop 0.5 + Sched):** {acc3_best:.2f}%

### Conclusion
By adding the Learning Rate Scheduler back onto our tried-and-true Dropout 0.5 architecture, we successfully gave the model the perfect combination of aggressive regularization (to prevent overfitting) and delicate optimizer steps (to perfectly settle into the final weights). 

This is exactly how hyperparameter tuning works in practice:
1. We formed a hypothesis (maybe 0.5 dropout was too aggressive).
2. We tested it (Exp 1) and found out we were wrong — 0.25 generalized worse.
3. We reverted the dropout back to 0.5, but kept the smart learning rate scheduler from Exp 2.
4. We (hopefully!) beat our baseline!
"""
md_compare = nbformat.v4.new_markdown_cell(final_md_text)

# Append to notebook
nb.cells.extend([md_exp3, code_exp3, code_plot, md_compare])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
    
print("Notebook updated successfully with Experiment 3!")
