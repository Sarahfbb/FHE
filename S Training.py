import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Configuration
EPOCHS = 10
BATCH_SIZE = 64

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-extracted features
train_features = np.load('train_features.npy')
train_labels = np.load('train_targets.npy')
val_features = np.load('val_features.npy')
val_labels = np.load('val_targets.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_targets.npy')

# Standardize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self.init_weights()

    def int_weights(self):
        int.xavier_uniform_(self.fc1.weight)
        int.xavier_uniform_(self.fc2.weight)
        int.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    


# Instantiate MLP
input_dim = train_features.shape[1]
num_classes = 10
mlp = MLPClassifier(input_dim, num_classes).to(device)

# Define optimizer and loss function for MLP
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# Convert features to torch tensors
train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long).to(device)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)

# Create DataLoader for MLP training
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader_mlp = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train the MLP model
def train_mlp(model, train_loader, device, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')
        
        # Save the model after each epoch
        model_path = f"mlp_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

train_mlp(mlp, train_loader_mlp, device, criterion, optimizer, epochs=EPOCHS)

# Evaluate the MLP model
def evaluate_mlp(model, features, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100. * correct / len(labels)
    return accuracy

train_accuracy_mlp = evaluate_mlp(mlp, train_features_tensor, train_labels_tensor)
val_accuracy_mlp = evaluate_mlp(mlp, val_features_tensor, val_labels_tensor)
test_accuracy_mlp = evaluate_mlp(mlp, test_features_tensor, test_labels_tensor)
print(f"MLP Classifier - Train Accuracy: {train_accuracy_mlp:.2f}%, Val Accuracy: {val_accuracy_mlp:.2f}%, Test Accuracy: {test_accuracy_mlp:.2f}%")

# Save the MLP model for quantization
torch.save(mlp.state_dict(), 'mlp_model.pth')