import nbformat
import json

notebook_path = "notebooks/main.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace("../model/final_predictions.csv", "../submission/final_predictions.csv")
    elif cell.cell_type == 'code':
        cell.source = cell.source.replace("../model/final_predictions.csv", "../submission/final_predictions.csv")

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Fixed paths in main.ipynb!")
