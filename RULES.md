# Digit Recognizer — Project Rules & Structure

## Directory Structure

```
digit-recognizer/
│
├── data/               # MNIST dataset lives here (auto-downloaded, never manually edited)
│
├── notebooks/          # Jupyter notebooks for exploration and visualisation
│                       # Use these to experiment, plot results, inspect feature maps
│
├── src/                # All main scripts — the real project lives here
│   ├── model.py        # CNN architecture definition
│   ├── train.py        # Training loop — runs model.py on MNIST
│   ├── evaluate.py     # Loads saved model, runs on test set, prints accuracy
│   └── predict.py      # Takes an image file as input, outputs a digit prediction
│
├── temp/               # Scratch space — throwaway scripts, quick tests, one-off experiments
│                       # Nothing here is part of the final project
│                       # Safe to delete at any time
│
├── model/              # Saved model weights go here after training
│   └── digit_cnn.pth   # PyTorch model checkpoint (created after running train.py)
│
└── RULES.md            # This file
```

---

## File Responsibilities

| File | What it does | When to run it |
|------|-------------|----------------|
| `src/model.py` | Defines the CNN class — layers, architecture | Never run directly, imported by others |
| `src/train.py` | Trains the model on MNIST, saves weights to `/model` | Once, at the start |
| `src/evaluate.py` | Measures accuracy on unseen test images | After training |
| `src/predict.py` | Predicts a single digit from an image file you provide | Anytime after training |

---

## Rules

### Structure rules
- **Single Notebook Workflow**: All code and explanations for every phase must be progressively built and documented in one comprehensive notebook (`notebooks/digit_recognizer_walkthrough.ipynb`). The user should be able to read this single file from top to bottom and understand every step of the project.
- While `src/` scripts can still be populated for standardisation, the primary development and teaching environment is the single notebook.
- All data goes in `/data` — never hardcode paths elsewhere, always reference `../data`
- All trained model weights go in `/model` — never save `.pth` files anywhere else
- Scripts that are experiments or helpers go in `/temp` — if it's not part of the core workflow, it goes there

### Code rules
- Every script must have a comment at the top explaining what it does and how to run it
- Every major block of code must have a comment explaining **why** it exists, not just what it does
  - Bad: `# apply max pooling`
  - Good: `# max pooling halves the feature map size, reducing computation in the next layer`
- No magic numbers — every hyperparameter must be named and explained
  ```python
  # Bad
  nn.Linear(1600, 128)

  # Good
  FLATTENED_SIZE = 1600   # 64 filters × 5×5 feature map after two rounds of pooling
  HIDDEN_SIZE = 128        # dense layer size, tunable — larger = more capacity, slower training
  nn.Linear(FLATTENED_SIZE, HIDDEN_SIZE)
  ```
- Imports go at the top of every file, grouped: standard library → third party → local

### Learning rules
- Every time a new concept appears in code (e.g. CrossEntropyLoss, DataLoader, optimizer.zero_grad),
  add a comment explaining what it is and why it is needed
- When a hyperparameter is set (learning rate, batch size, number of epochs), explain the reasoning
  in a comment — what happens if you increase or decrease it
- After training, always check which digits the model gets wrong most often — this tells you
  something about what the network is struggling with

### Workflow rules
- Always train first (`train.py`), then evaluate (`evaluate.py`), then predict (`predict.py`)
- Never import from `temp/` — if something in temp turns out to be useful, move it to `src/`
- Keep `/data` in `.gitignore` — MNIST is large and auto-downloads anyway
- Keep `/model` in `.gitignore` — weights are large and reproducible by retraining

---

## How the project builds up

```
Step 1 — src/model.py      define the CNN architecture
           ↓
Step 2 — src/train.py      train on MNIST, save weights to /model
           ↓
Step 3 — src/evaluate.py   load weights, test on unseen data, print accuracy
           ↓
Step 4 — src/predict.py    load weights, accept an image, print prediction
```

Each step builds on the previous one. You should be able to read each file
top to bottom and understand what is happening and why.

---

## Hyperparameter reference

These are the key numbers you will tune. Understand each one before changing it.

| Parameter | Location | What it controls |
|-----------|----------|-----------------|
| `LEARNING_RATE` | train.py | How big each weight update is — too high = unstable, too low = slow |
| `BATCH_SIZE` | train.py | How many images to process before updating weights — larger = faster but more memory |
| `EPOCHS` | train.py | How many full passes through the training data — more = better accuracy up to a point |
| `NUM_FILTERS_1` | model.py | Filters in first conv layer — more = detects more edge patterns |
| `NUM_FILTERS_2` | model.py | Filters in second conv layer — more = detects more complex shapes |
| `HIDDEN_SIZE` | model.py | Neurons in the dense layer — more = more classification capacity |