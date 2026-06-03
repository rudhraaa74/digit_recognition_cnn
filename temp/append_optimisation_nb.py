import nbformat

notebook_path = "notebooks/digit_recognizer_walkthrough.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Phase 8 Markdown
md_phase8 = nbformat.v4.new_markdown_cell("""---

# Phase 8: Optimization & Final Model Training

In our experiments, we discovered that the best way to push our model's accuracy to its absolute limit is to combine three powerful techniques:
1. **Aggressive Regularization**: Keeping our `Dropout` at `0.5` prevents the model from memorizing the data.
2. **Data Augmentation**: We dynamically rotate (±10 degrees) and shift (±10%) the images during training. This artificially creates an infinite amount of new data, forcing the model to learn robust geometric concepts rather than raw pixels.
3. **Learning Rate Scheduler**: We use `ReduceLROnPlateau` to automatically cut our learning rate in half when the model stops improving. This allows the optimizer to take tiny, delicate steps at the very end to settle perfectly into the final weights.

Because the task is much harder now (due to augmentation), we will train for **15 epochs** instead of 10.""")

# Phase 8 Code
code_phase8 = nbformat.v4.new_code_cell("""import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("\\n--- Starting Phase 8: Final Optimized Training ---")

# 1. Initialize our original model (which uses 0.5 Dropout by default)
final_model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_final = optim.Adam(final_model.parameters(), lr=0.001)

# 2. Setup the Scheduler
scheduler_final = ReduceLROnPlateau(optimizer_final, mode='max', patience=2, factor=0.5)

# 3. Setup Data Augmentation
augment = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1))
])

FINAL_EPOCHS = 15
FINAL_SAVE_PATH = "../model/digit_cnn_final.pth"
best_final_acc = 0.0

for epoch in range(1, FINAL_EPOCHS + 1):
    current_lr = optimizer_final.param_groups[0]['lr']
    
    # --- TRAINING PHASE ---
    final_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Apply data augmentation on the fly to the training batch
        images = augment(images)
        
        optimizer_final.zero_grad()
        loss = criterion(final_model(images), labels)
        loss.backward()
        optimizer_final.step()
        
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
        
    # --- VALIDATION PHASE ---
    final_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            
            # Never augment validation data!
            _, predicted = torch.max(final_model(val_images).data, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
            
    val_acc = 100.0 * correct / total
    
    # Step the scheduler
    scheduler_final.step(val_acc)
    
    # Print stats
    print(f"Epoch {epoch:2d}/{FINAL_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_final_acc:
        best_final_acc = val_acc
        torch.save(final_model.state_dict(), FINAL_SAVE_PATH)
        print(f"   --> New Best! Saved to {FINAL_SAVE_PATH}")

print(f"\\nFinal Optimization Complete! Best Validation Accuracy: {best_final_acc:.2f}%")""")

# Append to notebook
nb.cells.extend([md_phase8, code_phase8])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
    
print("Main Walkthrough Notebook updated successfully with Phase 8!")
