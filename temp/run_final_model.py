import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_dataloaders
from src.model import DigitCNN

def train_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Using DigitCNN (which has Dropout 0.5 built in)
    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    augment = T.Compose([
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    
    train_loader, val_loader = get_dataloaders(csv_path="data/digi_rec_train.csv", batch_size=64)
    
    best_acc = 0.0
    FINAL_EPOCHS = 15
    
    print("\\n--- Training Final Optimized Model ---")
    for epoch in range(1, FINAL_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = augment(images)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                _, predicted = torch.max(model(val_images).data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
                
        val_acc = 100.0 * correct / total
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model/digit_cnn_final.pth")
            
        print(f"Epoch {epoch} | Val Acc: {val_acc:.2f}%")
        
    print(f"\\nFinal Best Validation Accuracy: {best_acc:.2f}%")
    
    print("\\n--- Generating Kaggle Test Predictions ---")
    
    # Load best model for predictions
    model.load_state_dict(torch.load("model/digit_cnn_final.pth"))
    model.eval()
    
    # Since data/digi_rec_test.csv might have a header 'pixel0, pixel1...' we read it safely
    try:
        test_df = pd.read_csv("data/digi_rec_test.csv")
    except:
        print("Failed to read test csv")
        return
        
    test_pixels = test_df.values
    predictions = []
    
    print(f"Loaded {len(test_pixels)} test images. Running inference...")
    
    with torch.no_grad():
        for i in range(len(test_pixels)):
            pixel_row = test_pixels[i] / 255.0
            pixel_row = (pixel_row - 0.1307) / 0.3081
            image_tensor = torch.tensor(pixel_row, dtype=torch.float32).reshape(1, 1, 28, 28).to(device)
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            predictions.append({"ImageId": i + 1, "Label": predicted.item()})
            
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv("model/final_predictions.csv", index=False)
    print(f"Saved {len(predictions)} predictions to model/final_predictions.csv!")

if __name__ == "__main__":
    train_and_predict()
