import nbformat

notebook_path = "notebooks/digit_recognizer_walkthrough.ipynb"

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Create new cells for Phase 6
markdown_phase6 = nbformat.v4.new_markdown_cell("""---

# Phase 6: Prediction on a Single Image

Now it's your turn! You can draw your own digit and see if the model can guess it.
I have written a complete script in `src/predict.py` that handles this for you.

### How to test your own handwriting:
1. Open MS Paint, Preview, or any drawing tool.
2. Draw a digit (0-9) as large and centered as possible. (It's okay if you draw black text on a white background, the script will automatically invert it for you!)
3. Save it as a PNG file, e.g., `my_digit.png`, in the main project folder.
4. Run the code cell below, or run this in your terminal: `python3 src/predict.py my_digit.png`""")

code_predict = nbformat.v4.new_code_cell("""# Replace 'my_digit.png' with the actual path to your image once you've drawn it!
# For now, it might error if the file doesn't exist yet.

!python3 ../src/predict.py my_digit.png""")

# Append cells to notebook
nb.cells.extend([markdown_phase6, code_predict])

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Appended Phase 6 cells to notebook successfully.")
