import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
# Configuration
EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3  # Slightly increased learning rate
PATIENCE = 10  # For early stopping

# Start timing
start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class FHEFriendlyMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FHEFriendlyMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Added dropout for regularization
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def evaluate_mlp(model, features, labels, device):
    model.eval()
    with torch.no_grad():
        outputs = model(features.to(device))
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels.to(device)).sum().item()
        accuracy = 100. * correct / len(labels)
    return accuracy

def train_mlp(model, train_loader, val_features, val_labels, device, criterion, optimizer, scheduler, epochs, patience):
    best_val_accuracy = 0
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}'})

        val_accuracy = evaluate_mlp(model, val_features, val_labels, device)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
        
        # Step the scheduler with the validation accuracy
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model, 'fhe_friendly_mlp_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

    model.load_state_dict(best_model)
    return model

def main():
    # Load and preprocess data
    features_dir = 'features'
    train_features = np.load(os.path.join(features_dir, 'Train_Features.npy'))
    train_targets = np.load(os.path.join(features_dir, 'Train_Targets.npy'))
    val_features = np.load(os.path.join(features_dir, 'Val_Features.npy'))
    val_targets = np.load(os.path.join(features_dir, 'Val_Targets.npy'))
    test_features = np.load(os.path.join(features_dir, 'Test_Features.npy'))
    test_targets = np.load(os.path.join(features_dir, 'Test_Targets.npy'))
    
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    # Convert to PyTorch tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.long)
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.long)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.long)

    # Create DataLoader for training
    train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate MLP
    input_dim, num_classes = train_features.shape[1], 10
    mlp = FHEFriendlyMLPClassifier(input_dim, num_classes).to(device)


    # Define optimizer, loss function, and learning rate scheduler for MLP
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    # Train the MLP
    print("Training MLP...")
    train_mlp(mlp, train_loader, val_features_tensor, val_targets_tensor, device, criterion, optimizer, scheduler, epochs=EPOCHS, patience=PATIENCE)

    # Final evaluation
    train_accuracy = evaluate_mlp(mlp, train_features_tensor, train_targets_tensor, device)
    val_accuracy = evaluate_mlp(mlp, val_features_tensor, val_targets_tensor, device)
    test_accuracy = evaluate_mlp(mlp, test_features_tensor, test_targets_tensor, device)
    print(f"Final Accuracies - Train: {train_accuracy:.2f}%, Val: {val_accuracy:.2f}%, Test: {test_accuracy:.2f}%")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()