import nbformat

# Load the notebook
notebook_path = "notebooks/digit_recognizer_walkthrough.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Go through the cells and update them
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        # Fix data paths
        if "csv_path='../data/" in cell.source:
            cell.source = cell.source.replace("csv_path='../data/", "csv_path='data/")
            
        # Fix model paths
        if "../model/digit_cnn.pth" in cell.source:
            cell.source = cell.source.replace("../model/digit_cnn.pth", "model/digit_cnn.pth")
            
        if 'os.makedirs("../model", exist_ok=True)' in cell.source:
            cell.source = cell.source.replace('os.makedirs("../model", exist_ok=True)', 'os.makedirs("model", exist_ok=True)')
            
        # Replace the manual definition of DigitCNN with an import
        if "class DigitCNN(nn.Module):" in cell.source:
            cell.source = """import torch.nn as nn
import torch.nn.functional as F
from src.model import DigitCNN

print("DigitCNN class imported successfully from src.model!")"""

        # Also, training loop inside the notebook could be completely replaced by the actual src.train call,
        # but since the notebook is a walkthrough, let's keep the training loop visible there but with fixed paths,
        # OR we can just import the train function. Since they say "do the things you are doing in the notebook",
        # let's just make sure the notebook has the correct paths and executes perfectly.

# Save the notebook without outputs (clearing outputs so it can be re-run freshly)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
