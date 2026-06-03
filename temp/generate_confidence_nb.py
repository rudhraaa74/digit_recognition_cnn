import nbformat
import os

notebook_path = "notebooks/confidence.ipynb"

# Create a new notebook
nb = nbformat.v4.new_notebook()

# Markdown: Intro
markdown_intro = nbformat.v4.new_markdown_cell("""# Phase 6: Deep Dive into Test Predictions & Model Confidence

In this notebook, we'll dive deep into our CNN's "confidence". Softmax confidence means the probability the model assigns to its prediction.
- **A 95% confidence does NOT mean the model is right 95% of the time.** It just means the model's internal math assigned 95% probability to that class based on the features it saw.
- **When confidence is low (e.g. 50%)**, it typically means the model sees features belonging to two different digits (like a loop for an 8 and a curve for a 3) and cannot make a definitive decision.

Let's load the confidence report our analysis script generated and see how certain our model really is!""")

# Code: Load data
code_load = nbformat.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the confidence report
report = pd.read_csv('../model/confidence_report.csv')

# Load the raw test images so we can display them
test_df = pd.read_csv('../data/digi_rec_test.csv')
test_pixels = test_df.values

print(f"Loaded {len(report)} predictions and images.")""")

# Markdown: Histogram
markdown_hist = nbformat.v4.new_markdown_cell("""## Confidence Histogram
What does the overall confidence distribution look like? 
A good model should be heavily skewed toward high confidence (mostly 95%+). If the model were guessing randomly, confidence would be spread evenly across all bins.""")

# Code: Histogram
code_hist = nbformat.v4.new_code_cell("""plt.figure(figsize=(10, 6))
sns.histplot(report['Confidence'], bins=50, kde=False, color='blue')
plt.title('Distribution of Model Confidence across 28,000 Test Images', fontsize=14, fontweight='bold')
plt.xlabel('Confidence Score (Probability)', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xlim(0, 1)

# Add a line to show where 90% confidence is
plt.axvline(0.90, color='red', linestyle='--', label='90% Confidence threshold')
plt.legend()
plt.show()

# Let's see the percentage of images > 90% confidence
high_conf = len(report[report['Confidence'] > 0.90]) / len(report) * 100
print(f"{high_conf:.2f}% of predictions have > 90% confidence!")""")

# Markdown: Average confidence per digit
markdown_bar = nbformat.v4.new_markdown_cell("""## Average Confidence per Digit
Which digits are the hardest for the model? Typically, digits that share visual features with others (like 4/9 or 3/8) have lower average confidences because the model gets "torn" between them more often.""")

# Code: Bar chart
code_bar = nbformat.v4.new_code_cell("""avg_conf_per_digit = report.groupby('PredictedLabel')['Confidence'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='PredictedLabel', y='Confidence', data=avg_conf_per_digit, palette='viridis')
plt.title('Average Confidence by Predicted Digit', fontsize=14, fontweight='bold')
plt.xlabel('Digit', fontsize=12)
plt.ylabel('Average Confidence', fontsize=12)
plt.ylim(0.9, 1.0) # Zoom in to see the differences
plt.show()""")

# Markdown: Most confident
markdown_most_confident = nbformat.v4.new_markdown_cell("""## The "Perfect" Digits
Let's look at the absolute most confident prediction for each digit (the images where the model was closest to 100% sure).""")

# Code: Most confident
code_most_confident = nbformat.v4.new_code_cell("""fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Most Confident Prediction per Digit', fontsize=16, fontweight='bold')

for digit in range(10):
    # Find the row with the highest confidence for this digit
    top_row = report[report['PredictedLabel'] == digit].nlargest(1, 'Confidence').iloc[0]
    
    # ImageId is 1-indexed, so we subtract 1 to get the array index
    idx = int(top_row['ImageId']) - 1
    conf = top_row['Confidence']
    
    # Reshape the 784 pixels back into 28x28
    img = test_pixels[idx].reshape(28, 28)
    
    r = digit // 5
    c = digit % 5
    ax = axes[r, c]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Digit {digit} | Conf: {conf*100:.2f}%", color='green')
    ax.axis('off')

plt.tight_layout()
plt.show()""")

# Markdown: Least confident
markdown_least_confident = nbformat.v4.new_markdown_cell("""## The 20 Most Uncertain Predictions
Now let's look at where the model struggled the most. These are the 20 images with the lowest overall confidence scores.
Notice how messy, ambiguous, or poorly written these digits are! This is why human handwriting is hard.""")

# Code: Least confident
code_least_confident = nbformat.v4.new_code_cell("""bottom_20 = report.nsmallest(20, 'Confidence')

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Top 20 Least Confident Predictions Overall', fontsize=16, fontweight='bold')

for i, (index, row) in enumerate(bottom_20.iterrows()):
    idx = int(row['ImageId']) - 1
    pred = int(row['PredictedLabel'])
    conf = row['Confidence']
    
    img = test_pixels[idx].reshape(28, 28)
    
    r = i // 5
    c = i % 5
    ax = axes[r, c]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pred: {pred} | Conf: {conf*100:.1f}%", color='red')
    ax.axis('off')

plt.tight_layout()
plt.show()""")

# Assemble notebook
nb.cells.extend([
    markdown_intro, code_load,
    markdown_hist, code_hist,
    markdown_bar, code_bar,
    markdown_most_confident, code_most_confident,
    markdown_least_confident, code_least_confident
])

# Ensure directory exists
os.makedirs("notebooks", exist_ok=True)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Generated {notebook_path} successfully.")
