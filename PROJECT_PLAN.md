# AI Instructions — Digit Recognizer Build

## How to use this file
These instructions are for an AI assistant helping build this project phase by phase.
Read this entire file before writing a single line of code.
Follow every rule strictly. Do not skip ahead.

---

## Golden Rules

- Complete one phase at a time. After each phase, stop and wait for the user to
  review and explicitly say "approved" or "move to next phase" before continuing.
- Never start the next phase without approval — even if the current phase looks complete.
- Always explain what you are doing and why as you do it. The user is learning.
  Do not just produce code — teach through the code.
- Every file you create must follow RULES.md exactly — correct folder, correct naming,
  comments on every major block explaining why not just what.
- If something is unclear or ambiguous, ask the user before assuming.

---

## Dataset Rules — Read Carefully

The train and test CSV files have already been downloaded from Kaggle and are located at:

  data/train.csv    — 42,000 labeled images for training and validation
  data/test.csv     — 28,000 images for final testing only

### Critical rules about the data:
- data/test.csv must NEVER be touched during training or validation.
  It exists only for the final evaluation in Phase 5.
- data/train.csv is the only file used during Phase 2, 3, and 4.
- During training, split data/train.csv into a training set (80%) and
  a validation set (20%) in code — do not create separate files for this.
- Do not download MNIST from torchvision. All data comes from the CSV files above.
- Do not modify, overwrite, or resave either CSV file at any point.

### CSV format (Kaggle MNIST):
- data/train.csv — first column is the label (0–9), remaining 784 columns are
  pixel values (pixel0 to pixel783), each value 0–255
- data/test.csv — 784 pixel columns only, no label column

---

## Phase Instructions

---

### Phase 1 — Environment Setup

Goal: verify the environment is ready before writing any real code.

Tasks:
1. Create the full folder structure as defined in RULES.md:
   digit-recognizer/data/, /notebooks/, /src/, /temp/, /model/
2. Write temp/verify.py that:
   - Imports torch, torchvision, matplotlib, pandas, numpy
   - Prints the torch version
   - Prints whether CUDA is available (GPU) or CPU will be used
   - Reads the first 5 rows of data/train.csv and prints the shape
     to confirm the data file is accessible
3. Tell the user exactly how to run it and what output to expect

Explain to the user:
- What each library does and why it is needed
- What CUDA is and why it does not matter for this project (MNIST trains fast on CPU)

Stop after Phase 1 and wait for approval.

---

### Phase 2 — Data Pipeline

Goal: load the Kaggle CSV data, understand its structure, prepare it for training.

Tasks:
1. Write src/dataset.py that:
   - Reads data/train.csv using pandas
   - Separates labels (first column) from pixel values (remaining 784 columns)
   - Normalises pixel values from 0–255 to 0.0–1.0 range
   - Applies standard MNIST normalisation: mean=0.1307, std=0.3081
   - Reshapes each row from 784 values into a 1×28×28 tensor (channel × height × width)
   - Splits into 80% train, 20% validation using a fixed random seed (seed=42)
   - Wraps both splits in PyTorch DataLoaders with batch_size=64
   - Exposes get_train_loader() and get_val_loader() functions
2. Write notebooks/explore_data.ipynb that:
   - Loads the DataLoader from src/dataset.py
   - Visualises 10 sample images in a grid with their labels
   - Prints the shape of one batch to confirm (64, 1, 28, 28)

Explain to the user:
- Why we normalise pixel values and what the mean/std values mean
- Why we reshape to 1×28×28 (the 1 is the channel — grayscale has 1, RGB has 3)
- What a DataLoader does and why we train in batches of 64 not one image at a time
- What a random seed is and why we use seed=42 (reproducibility)
- Why we split train.csv into train/val instead of using test.csv for validation

Stop after Phase 2 and wait for approval.

---

### Phase 3 — Model Architecture

Goal: define the CNN in code. The user must understand every layer before moving on.

Tasks:
1. Write src/model.py defining the DigitCNN class with this exact architecture:

   Input        (1 × 28 × 28)
       ↓
   Conv Layer 1    32 filters, 3×3 kernel, ReLU
       ↓           output: 32 × 26 × 26
   Max Pool 1      2×2
       ↓           output: 32 × 13 × 13
   Conv Layer 2    64 filters, 3×3 kernel, ReLU
       ↓           output: 64 × 11 × 11
   Max Pool 2      2×2
       ↓           output: 64 × 5 × 5
   Flatten         64 × 5 × 5 = 1600
       ↓
   Dense Layer     1600 → 128, ReLU
       ↓
   Dropout         p=0.5
       ↓
   Output Layer    128 → 10

2. Every single layer must have a comment block explaining:
   - What it does
   - Why it is in the architecture
   - What the input and output shape is
3. All hyperparameters (filter counts, hidden size, dropout rate) must be named
   constants at the top of the file with comments explaining what happens if you
   change them
4. Write temp/test_model.py that:
   - Creates a DigitCNN instance
   - Passes a fake tensor of shape (1, 1, 28, 28) through it
   - Prints the output shape — must be (1, 10)
   - Prints the total number of trainable parameters in the model

Explain to the user:
- How each conv layer shrinks the image size (why 28→26 after a 3×3 filter)
- What ReLU does and why every conv layer needs it
- What max pooling does (keeps the highest value in each 2×2 block)
- What dropout does and why p=0.5 means 50% of neurons are randomly switched off
- Why the output is 10 numbers and not a single digit prediction

Stop after Phase 3 and wait for approval.

---

### Phase 4 — Training Loop

Goal: train the model on the Kaggle train.csv data, save the best weights.

Tasks:
1. Write src/train.py that:
   - Imports DigitCNN from src/model.py
   - Imports DataLoaders from src/dataset.py
   - Defines these named constants with comments:
       LEARNING_RATE = 0.001
       EPOCHS = 10
       MODEL_SAVE_PATH = "model/digit_cnn.pth"
   - Uses CrossEntropyLoss as the loss function
   - Uses Adam optimizer
   - Runs the training loop for 10 epochs
   - After each epoch prints: epoch number, training loss, validation accuracy
   - Saves model weights to model/digit_cnn.pth only when validation accuracy improves
   - Never touches data/test.csv at any point
2. Tell the user exactly what to expect each epoch — roughly what accuracy numbers
   indicate the model is training correctly vs something being wrong

Explain to the user:
- What CrossEntropyLoss measures and why it is the right choice for classification
- What the Adam optimizer does (adapts learning rate per parameter automatically)
- What backpropagation is in plain english — loss flows backwards, weights adjust
- Why we track validation accuracy and not just training accuracy
- Why we only save when validation accuracy improves (not just the last epoch)
- What overfitting looks like in the printed numbers (train acc rising, val acc plateauing)

Expected output per epoch:
  Epoch 1  — val accuracy ~97%
  Epoch 5  — val accuracy ~98.5%
  Epoch 10 — val accuracy ~99%+

Stop after Phase 4 and wait for approval.

---

### Phase 5 — Evaluation on Test Data

Goal: measure real performance on data/test.csv which has never been seen before.

Tasks:
1. Write src/evaluate.py that:
   - Loads model weights from model/digit_cnn.pth
   - Loads data/test.csv (no labels — this is the Kaggle test set)
   - Applies the exact same preprocessing as src/dataset.py (normalise, reshape)
   - Runs inference on all 28,000 test images
   - Saves predictions to model/predictions.csv with columns: ImageId, Label
   - Also evaluates on the validation split from train.csv (which has labels)
     and prints accuracy, so the user can see a real accuracy number
2. Write notebooks/evaluate.ipynb that:
   - Loads the validation set predictions
   - Shows a confusion matrix as a heatmap
   - Visualises 10 images the model got wrong with the true label and predicted label

Explain to the user:
- Why data/test.csv has no labels (it is the Kaggle competition test set)
- What a confusion matrix tells you (which digits get confused with which)
- Which digit pairs are typically hardest (4/9, 3/8, 1/7) and why
- What the predictions.csv file is for (submitting to Kaggle if desired)

Stop after Phase 5 and wait for approval.

---

### Phase 6 — Deep Dive into Test Predictions

Goal: analyse the model's confidence across all 28,000 Kaggle test predictions to understand where it is certain and where it struggles.

Tasks:
1. Write src/analyse.py that:
   - Loads model/digit_cnn.pth
   - Loads data/test.csv and applies the same preprocessing as training
   - Runs inference on all 28,000 images and applies softmax to get probabilities
   - Finds the top 10 most confident predictions (closest to 100%) for each digit 0–9
   - Finds the top 20 least confident predictions overall (model was most uncertain)
   - Saves a summary to model/confidence_report.csv with columns: ImageId, PredictedLabel, Confidence

2. Write notebooks/confidence.ipynb that:
   - Plots a histogram of confidence scores across all 28,000 predictions — shows how often the model is 99%+ confident vs below 80%
   - Displays the 20 least confident predictions as a grid with the predicted digit and confidence percentage shown under each image
   - Displays the most confident prediction for each digit 0–9 in a row
   - Plots a bar chart showing average confidence per digit — which digits does the model predict most and least confidently

Explain to the user:
- What softmax confidence actually means (a 95% confidence does not mean the model is right 95% of the time — it means it assigned 95% probability to that class)
- What it means when confidence is low — the model sees features of two digits and cannot decide
- Which digits typically have lower average confidence and why (visually similar pairs like 4/9, 3/8)
- What the confidence histogram shape tells you about the model overall — a good model should be heavily skewed toward high confidence

Stop after Phase 6 and wait for approval.

### Phase 7 — Experiments (optional)

Goal: build intuition by deliberately breaking and improving the model.

Only begin this phase if the user explicitly asks for it.

Suggested experiments (all in temp/, never in src/):
- temp/exp_no_dropout.py     — remove dropout, observe overfitting
- temp/exp_deeper.py         — add a third conv layer, check if accuracy improves
- temp/exp_lr.py             — change learning rate to 0.01, observe instability
- temp/exp_small_dense.py    — reduce dense layer from 128 to 32 neurons

For each experiment:
- Run it and compare validation accuracy to the baseline
- Explain to the user what changed and why the result makes sense

---

## What good output looks like

At every phase the AI should:
- Explain what it is about to do before doing it
- Write the code with comments that teach, not just describe
- After writing each file, explain what the user should see when they run it
- Flag anything the user should pay attention to or might find confusing

The user is learning. Every decision in the code is a teaching opportunity.
Do not rush. Do not skip explanations. Accuracy of understanding matters
as much as accuracy of the model.