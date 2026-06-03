import sys
import os
import torch

# Add the parent directory to the path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DigitCNN

def test_model():
    print("--- Testing DigitCNN Architecture ---")
    
    # 1. Create an instance of our model
    model = DigitCNN()
    
    # 2. Create a fake image tensor
    # Shape: (Batch Size=1, Channels=1, Height=28, Width=28)
    dummy_input = torch.randn(1, 1, 28, 28)
    print(f"Input shape: {dummy_input.shape}")
    
    # 3. Pass the fake image through the model (forward pass)
    output = model(dummy_input)
    
    # 4. Check the output shape
    # It must be (1, 10) because we have 1 image and 10 possible digits (0-9)
    print(f"Output shape: {output.shape}")
    if output.shape == (1, 10):
        print("✅ Output shape is correct!")
    else:
        print("❌ Output shape is incorrect!")
    
    # 5. Calculate and print total trainable parameters
    # This helps us understand how "big" or complex the model is.
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

if __name__ == "__main__":
    test_model()
