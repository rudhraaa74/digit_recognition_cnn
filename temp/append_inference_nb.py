import nbformat

notebook_path = "notebooks/main.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Phase 8 Inference Markdown
md_inference = nbformat.v4.new_markdown_cell("""---

## Generating Kaggle Predictions

Now that our final model is fully trained and the weights are saved, we need to generate predictions on the unlabeled Kaggle test set (`data/digi_rec_test.csv`). We will process all 28,000 images exactly as we did during training, and save the output to `../model/final_predictions.csv` so it is ready to upload to the Kaggle leaderboard!""")

# Phase 8 Inference Code
code_inference = nbformat.v4.new_code_cell("""import pandas as pd

print("\\n--- Generating Kaggle Test Predictions ---")

# 1. Load the best model weights we just saved
final_model.load_state_dict(torch.load(FINAL_SAVE_PATH))
final_model.eval() # Disable dropout for inference!

# 2. Load the Kaggle test CSV (no labels, only pixels)
test_df = pd.read_csv("../data/digi_rec_test.csv")
test_pixels = test_df.values

predictions = []

print(f"Loaded {len(test_pixels)} test images. Running inference...")

# We don't need gradients for inference
with torch.no_grad():
    for i in range(len(test_pixels)):
        # Preprocess exactly the same as training
        pixel_row = test_pixels[i] / 255.0
        pixel_row = (pixel_row - 0.1307) / 0.3081
        
        # Reshape and convert to Tensor
        image_tensor = torch.tensor(pixel_row, dtype=torch.float32).reshape(1, 1, 28, 28).to(device)
        
        # Predict
        output = final_model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        
        # Save prediction (ImageId is 1-indexed for Kaggle)
        predictions.append({"ImageId": i + 1, "Label": predicted.item()})
        
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} images...")
            
# 3. Save to CSV
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("../model/final_predictions.csv", index=False)
print(f"\\nSaved {len(predictions)} predictions to ../model/final_predictions.csv!")
print("This file is 100% ready to upload to Kaggle!")""")

# Append to notebook
nb.cells.extend([md_inference, code_inference])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
    
print("Notebook updated successfully with inference code!")
