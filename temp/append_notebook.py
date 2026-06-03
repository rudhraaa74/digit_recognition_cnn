import nbformat
import sys

notebook_path = "notebooks/digit_recognizer_walkthrough.ipynb"

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Create new cells for Phase 5
markdown_phase5 = nbformat.v4.new_markdown_cell("""---

# Phase 5: Evaluation & Confusion Matrix

Let's evaluate the model on the validation set, generate a confusion matrix to see which digits get confused with which, and look at the images the model gets completely wrong.""")

code_confusion = nbformat.v4.new_code_cell("""import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import get_val_loader
from src.model import DigitCNN

# 1. Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN().to(device)
model.load_state_dict(torch.load("../model/digit_cnn.pth", map_location=device))
model.eval()

# 2. Collect all predictions and true labels
val_loader = get_val_loader(csv_path="../data/digi_rec_train.csv", batch_size=128)
all_preds = []
all_labels = []
wrong_images = []
wrong_preds = []
wrong_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Save misclassified images
        wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
        for idx in wrong_idx:
            if len(wrong_images) < 10:
                wrong_images.append(images[idx].cpu())
                wrong_preds.append(preds[idx].item())
                wrong_labels.append(labels[idx].item())

# 3. Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix on Validation Set', fontsize=14, fontweight='bold')
plt.show()""")

markdown_mistakes = nbformat.v4.new_markdown_cell("""Notice that some digit pairs are typically harder to distinguish than others! For example:
- 4 and 9 (they look very similar if the loop on the 9 is sloppy)
- 3 and 8 
- 1 and 7

Let's look at 10 examples where the model got it wrong!""")

code_mistakes = nbformat.v4.new_code_cell("""# Plot 10 images the model got wrong
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Oops! Misclassified Images", fontsize=16, fontweight='bold', y=1.05)

for i, ax in enumerate(axes.flat):
    if i < len(wrong_images):
        image_2d = wrong_images[i].squeeze().numpy()
        ax.imshow(image_2d, cmap='gray')
        ax.set_title(f"True: {wrong_labels[i]} | Pred: {wrong_preds[i]}", color='red')
    ax.axis('off')

plt.tight_layout()
plt.show()""")

# Append cells to notebook
nb.cells.extend([markdown_phase5, code_confusion, markdown_mistakes, code_mistakes])

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Appended Phase 5 cells to notebook successfully.")
