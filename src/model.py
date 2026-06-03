"""
Phase 3: Model Architecture

This script defines our Convolutional Neural Network (CNN) for digit recognition.
"""

import torch
import torch.nn as nn

# --- Hyperparameters ---
# You can tweak these values to experiment with the model's capacity and regularisation.

# Number of filters in the first convolutional layer.
# Increasing this lets the model learn more low-level features (like edges and curves), 
# but makes the model slower and more prone to overfitting.
CONV1_FILTERS = 32

# Number of filters in the second convolutional layer.
# Increasing this lets the model learn more complex combinations of the low-level features.
CONV2_FILTERS = 64

# Number of neurons in the fully connected (dense) hidden layer.
# A larger number allows the network to learn more complex relationships before the final classification,
# but significantly increases the parameter count.
HIDDEN_SIZE = 128

# Probability of dropping a neuron during training in the Dropout layer.
# Increasing this (e.g., to 0.7) adds more regularisation, making the model learn more robust features
# and preventing overfitting, but might make it harder to train if set too high.
DROPOUT_RATE = 0.5


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        
        # --- 1st Convolutional Block ---
        # What it does: Slides 32 small 3x3 filters over the image to detect simple features like edges.
        # Why it is here: CNNs need to find local patterns in an image, irrespective of exactly where they are.
        # Input shape:  (BatchSize, 1, 28, 28) - 1 channel (grayscale)
        # Output shape: (BatchSize, 32, 26, 26) - 32 feature maps. Size shrinks from 28 to 26 because we don't pad the edges.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV1_FILTERS, kernel_size=3)
        self.relu1 = nn.ReLU()
        
        # What it does: Reduces the spatial size of the image by taking the maximum value in every 2x2 window.
        # Why it is here: Reduces computation and makes the feature detection less sensitive to small shifts in position.
        # Input shape:  (BatchSize, 32, 26, 26)
        # Output shape: (BatchSize, 32, 13, 13) - Width and height are halved.
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # --- 2nd Convolutional Block ---
        # What it does: Slides 64 3x3 filters over the 32 feature maps from the previous layer.
        # Why it is here: Detects higher-level, more complex features (e.g. loops, intersections) built from simpler edges.
        # Input shape:  (BatchSize, 32, 13, 13)
        # Output shape: (BatchSize, 64, 11, 11) - Size shrinks from 13 to 11.
        self.conv2 = nn.Conv2d(in_channels=CONV1_FILTERS, out_channels=CONV2_FILTERS, kernel_size=3)
        self.relu2 = nn.ReLU()
        
        # What it does: Halves the spatial dimensions again.
        # Why it is here: Further compresses the representation and controls overfitting.
        # Input shape:  (BatchSize, 64, 11, 11)
        # Output shape: (BatchSize, 64, 5, 5) - Width and height are halved (11/2 = 5, rounded down).
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # --- Fully Connected (Dense) Block ---
        # The output of pool2 is a 3D volume (64 x 5 x 5). We need to flatten it into a 1D array to pass into a Dense layer.
        # 64 channels * 5 height * 5 width = 1600 features.
        
        # What it does: A standard neural network layer connecting all 1600 incoming features to 128 hidden neurons.
        # Why it is here: Combines all the spatial features found by the CNN into a high-level representation to make a decision.
        # Input shape:  (BatchSize, 1600)
        # Output shape: (BatchSize, 128)
        self.fc1 = nn.Linear(in_features=CONV2_FILTERS * 5 * 5, out_features=HIDDEN_SIZE)
        self.relu3 = nn.ReLU()
        
        # What it does: Randomly zeroes out 50% of the neurons during training.
        # Why it is here: Prevents the network from relying too heavily on any single feature (overfitting), forcing it to learn redundant representations.
        # Input shape:  (BatchSize, 128)
        # Output shape: (BatchSize, 128)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        
        # What it does: Connects the 128 hidden neurons to the 10 final output classes (digits 0-9).
        # Why it is here: Gives us our final predictions.
        # Input shape:  (BatchSize, 128)
        # Output shape: (BatchSize, 10) - 10 raw scores (logits), one for each digit.
        self.fc2 = nn.Linear(in_features=HIDDEN_SIZE, out_features=10)

    def forward(self, x):
        """
        Defines the forward pass of the model. 
        This is the path data takes when we pass it into our network.
        """
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        # x.view() reshapes the tensor. 
        # x.size(0) keeps the batch size, and -1 tells PyTorch to figure out the remaining dimension (1600).
        x = x.view(x.size(0), -1) 
        
        # Fully Connected Block
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        
        # Output
        x = self.fc2(x)
        
        return x
