import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import src.config as cfg
from src.data_prep.dataset import AcousticMachineryDataset
from src.models.cnn_classifier import AcousticAnomalyCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training on: {device}")

    # Load Dataset
    dataset = AcousticMachineryDataset(data_dir=cfg.RAW_DATA_DIR)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Init Model, Loss, Optimizer
    model = AcousticAnomalyCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Training Loop
    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {val_acc:.2f}%")

    print("Training Complete. Saving model weights...")
    torch.save(model.state_dict(), "acoustic_cnn_weights.pth")

if __name__ == "__main__":
    # Ensure you have a 'data/raw/healthy' and 'data/raw/broken' folder with .wav files before running!
    train()
