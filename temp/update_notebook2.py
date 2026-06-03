import nbformat

# Load the notebook
notebook_path = "notebooks/digit_recognizer_walkthrough.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Go through the cells and update them back to ../ paths 
for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        # Fix data paths
        if "csv_path='data/" in cell.source:
            cell.source = cell.source.replace("csv_path='data/", "csv_path='../data/")
            
        # Fix model paths
        if "model/digit_cnn.pth" in cell.source and "../model" not in cell.source:
            cell.source = cell.source.replace("model/digit_cnn.pth", "../model/digit_cnn.pth")
            
        if 'os.makedirs("model", exist_ok=True)' in cell.source:
            cell.source = cell.source.replace('os.makedirs("model", exist_ok=True)', 'os.makedirs("../model", exist_ok=True)')

# Save the notebook 
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
