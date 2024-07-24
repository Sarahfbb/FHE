
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler
from imblearn.over_sampling import RandomOverSampler

# Configuration
BATCH_SIZE = 256  # Reduced from 1024
EPOCHS = 1 
MODELS_PER_CLASS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_SAMPLES_AFTER_OVERSAMPLING = 100000  # New parameter to limit oversampled data size

# Start timing
start_time = time.time()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FHEFriendlyMLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FHEFriendlyMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.poly_activation(self.fc1(x))
        x = self.poly_activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def poly_activation(self, x):
        return 0.5 * x + 0.25 * x**2

def create_binary_labels_and_apply_oversampling(features, targets, class_id, max_samples=MAX_SAMPLES_AFTER_OVERSAMPLING):
    binary_labels = np.where(targets == class_id, 1, 0)
    ros = RandomOverSampler(random_state=42, sampling_strategy={0: min(sum(binary_labels == 0), max_samples // 2), 
                                                                1: min(sum(binary_labels == 1), max_samples // 2)})
    features_resampled, labels_resampled = ros.fit_resample(features, binary_labels)
    return features_resampled, labels_resampled

def train_ensemble(train_features, train_targets, models_per_class=MODELS_PER_CLASS, batch_size=BATCH_SIZE, epochs=EPOCHS):
    all_ensembles = {}
    input_dim = train_features.shape[1]
    grad_scaler = GradScaler()
    
    for class_id in range(7,10):
        print(f"Training ensemble for class {class_id}")
        
        class_ensemble = []
        for i in range(MODELS_PER_CLASS):
            bootstrap_indices = np.random.choice(len(train_features), size=len(train_features), replace=True)
            bootstrap_features = train_features[bootstrap_indices]
            bootstrap_targets = train_targets[bootstrap_indices]
            
            resampled_features, resampled_targets = create_binary_labels_and_apply_oversampling(bootstrap_features, bootstrap_targets, class_id)
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(resampled_features)
            
            dataset = TensorDataset(torch.FloatTensor(normalized_features), torch.FloatTensor(resampled_targets).unsqueeze(1))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            
            model = FHEFriendlyMLPClassifier(input_dim).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch_features, batch_labels in dataloader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            print(f"Class {class_id}, Model {i+1}/{models_per_class}, Loss: {avg_loss:.4f}")

            class_ensemble.append(model)
            torch.save(model.state_dict(), f'fhe_friendly_ensemble_class_{class_id}_mlp_{i}.pt')

            torch.cuda.empty_cache()

        all_ensembles[class_id] = class_ensemble

    return all_ensembles

if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # Warm-up CUDA
    dummy_tensor = torch.zeros(1, device=device)
    dummy_tensor = dummy_tensor + 1
    
    # Load pre-extracted features
    train_features = np.load('train_features.npy')
    train_targets = np.load('train_targets.npy')

    print(f"Loaded data shapes:")
    print(f"Train features: {train_features.shape}, Train targets: {train_targets.shape}")
    
    all_class_ensembles = train_ensemble(train_features, train_targets)
    print("All ensemble models trained and saved.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time:.2f} seconds")