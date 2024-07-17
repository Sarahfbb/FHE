import time
import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
i
# Configuration
BATCH_SIZE = 32
EPOCHS = 5
MODELS_PER_CLASS = 10

# Start timing
start_time = time.time()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-extracted features (keep on CPU)
train_features = np.load('train_features.npy')
train_targets = np.load('train_targets.npy')

print(f"Loaded data shapes:")
print(f"Train features: {train_features.shape}, Train targets: {train_targets.shape}")

import torch.nn.init as init

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_binary_labels_and_apply_oversampling(features, targets, class_id):
    binary_labels = np.where(targets == class_id, 1, 0)
    ros = RandomOverSampler(random_state=42)
    features_resampled, labels_resampled = ros.fit_resample(features, binary_labels)
    return features_resampled, labels_resampled

def train_ensemble(train_features, train_targets, models_per_class=MODELS_PER_CLASS, batch_size=BATCH_SIZE, epochs=EPOCHS):
    all_ensembles = {}
    input_dim = train_features.shape[1]
    scaler = GradScaler()
    
    for class_id in range(0, 10):  # Changed from range(10) to range(4, 10)
        print(f"Training ensemble for class {class_id}")
        
        class_ensemble = []
        for i in range(models_per_class):
            print(f"Training model {i+1}/{models_per_class} for class {class_id}")
            
            bootstrap_indices = np.random.choice(len(train_features), size=len(train_features), replace=True)
            bootstrap_features = train_features[bootstrap_indices]
            bootstrap_targets = train_targets[bootstrap_indices]
            
            resampled_features, resampled_targets = create_binary_labels_and_apply_oversampling(bootstrap_features, bootstrap_targets, class_id)
            
            # Normalize features batch-wise to save memory
            dataset = TensorDataset(torch.FloatTensor(resampled_features), torch.FloatTensor(resampled_targets).unsqueeze(1))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            
            model = MLPClassifier(input_dim).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Reduced learning rate

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                batch_count = 0
                
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                
                for batch_features, batch_labels in dataloader:
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    
                    # Normalize batch
                    mean = batch_features.mean(dim=0)
                    std = batch_features.std(dim=0)
                    batch_features = (batch_features - mean) / (std + 1e-8)
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_labels)

                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    batch_count += 1

                end_event.record()
                torch.cuda.synchronize()
                
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/batch_count:.4f}")

            class_ensemble.append(model)
            torch.save(model.state_dict(), f'ensemble_class_{class_id}_mlp_{i}.pt')

            # Free memory
            del bootstrap_features, bootstrap_targets, resampled_features, resampled_targets
            torch.cuda.empty_cache()

        all_ensembles[class_id] = class_ensemble

    return all_ensembles

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # Warm-up CUDA
    dummy_tensor = torch.zeros(1, device=device)
    dummy_tensor = dummy_tensor + 1
    
    all_class_ensembles = train_ensemble(train_features, train_targets)
    print("All ensemble models trained and saved.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time:.2f} seconds")
