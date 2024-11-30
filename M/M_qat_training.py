import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os

# Brevitas imports
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.nn import QuantIdentity, QuantLinear

# Configuration
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BIT_WIDTH = 6
NUM_CLASSES = 10

print(f"Using device: {DEVICE}")

class QuantBatchNorm1d(nn.Module):
    def __init__(self, num_features, bit_width=BIT_WIDTH):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        x = self.bn(x.value)
        return self.quant(x)

class QuantFHEFriendlyBinaryClassifier(nn.Module):
    def __init__(
        self, 
        input_dim, 
        bit: int = BIT_WIDTH,
        act_quant: nn.Module = Int8ActPerTensorFloat,
        weight_quant: nn.Module = Int8WeightPerTensorFloat,
    ):
        super(QuantFHEFriendlyBinaryClassifier, self).__init__()
        
        self.input_quant = QuantIdentity(bit_width=bit, act_quant=act_quant, return_quant_tensor=True)
        
        self.fc1 = QuantLinear(input_dim, 256, bias=True, weight_bit_width=bit, weight_quant=weight_quant, return_quant_tensor=True)
        self.bn1 = QuantBatchNorm1d(256, bit_width=bit)
        self.relu1 = qnn.QuantReLU(return_quant_tensor=True, bit_width=bit, act_quant=act_quant)
        
        self.fc2 = QuantLinear(256, 128, bias=True, weight_bit_width=bit, weight_quant=weight_quant, return_quant_tensor=True)
        self.bn2 = QuantBatchNorm1d(128, bit_width=bit)
        self.relu2 = qnn.QuantReLU(return_quant_tensor=True, bit_width=bit, act_quant=act_quant)
        
        self.fc3 = QuantLinear(128, 1, bias=True, weight_bit_width=bit, weight_quant=weight_quant, return_quant_tensor=True)
        
        self.dropout = qnn.QuantDropout(p=0.5)
        
    def forward(self, x):
        x = self.input_quant(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.value

class EnsembleQuantizedModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EnsembleQuantizedModel, self).__init__()
        self.models = nn.ModuleList([QuantFHEFriendlyBinaryClassifier(input_dim) for _ in range(num_classes)])

    def forward(self, x):
        outputs = torch.cat([model(x) for model in self.models], dim=1)
        return outputs

    def load_individual_models(self, device):
        for class_id in range(NUM_CLASSES):
            model_filename = f'fhe_friendly_binary_classifier_class_{class_id}.pth'
            if os.path.exists(model_filename):
                model_state = torch.load(model_filename, map_location=device)
                self.models[class_id].load_state_dict(model_state, strict=False)
                self.models[class_id].to(device)  # Ensure the model is on the correct device
                print(f"Loaded weights for class {class_id}")
            else:
                print(f"Warning: {model_filename} not found.")

def evaluate_ensemble(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train_qat(model, train_loader, val_loader, param, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_accuracy = 0
    for epoch in range(param['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{param['epochs']}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{running_loss/total:.4f}"})

        train_accuracy = 100 * correct / total
        val_accuracy = evaluate_ensemble(model, val_loader, device)

        scheduler.step(val_accuracy)

        param['accuracy_train'].append(train_accuracy)
        param['accuracy_test'].append(val_accuracy)
        param['loss_train_history'].append(running_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{param['epochs']}, Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'Best_Fhe_friendly_qat_ensemble_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

    # Save the final model
    torch.save(model.state_dict(), 'Final_Fhe_friendly_qat_ensemble_model.pth')
    return model

def load_and_preprocess_data(features_dir):
    print("Loading data...")
    train_features = np.load(os.path.join(features_dir, 'Train_Features.npy'))
    train_targets = np.load(os.path.join(features_dir, 'Train_Targets.npy'))
    val_features = np.load(os.path.join(features_dir, 'Val_Features.npy'))
    val_targets = np.load(os.path.join(features_dir, 'Val_Targets.npy'))
    test_features = np.load(os.path.join(features_dir, 'Test_Features.npy'))
    test_targets = np.load(os.path.join(features_dir, 'Test_Targets.npy'))
    print("Data loaded successfully.")

    print("Preprocessing data...")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    print("Data preprocessing completed.")

    return train_features, train_targets, val_features, val_targets, test_features, test_targets

def main():
    try:
        start_time = time.time()
        
        features_dir = 'features'
        train_features, train_targets, val_features, val_targets, test_features, test_targets = load_and_preprocess_data(features_dir)

        # Move data to the specified device
        train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32).to(DEVICE), 
                                      torch.tensor(train_targets, dtype=torch.long).to(DEVICE))
        val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32).to(DEVICE), 
                                    torch.tensor(val_targets, dtype=torch.long).to(DEVICE))
        test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32).to(DEVICE), 
                                     torch.tensor(test_targets, dtype=torch.long).to(DEVICE))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        print("DataLoaders created.")

        input_dim, num_classes = train_features.shape[1], len(np.unique(train_targets))
        qat_ensemble = EnsembleQuantizedModel(input_dim, num_classes).to(DEVICE)

        print("Loading pre-trained weights...")
        qat_ensemble.load_individual_models(DEVICE)
        print("Pre-trained weights loaded and mapped to the quantized model.")

        param = {
            'lr': LEARNING_RATE,
            'epochs': EPOCHS,
            'accuracy_train': [],
            'accuracy_test': [],
            'loss_train_history': [],
        }

        print("Training Quantized Ensemble...")
        train_qat(qat_ensemble, train_loader, val_loader, param, DEVICE)

        print("Loading best model for evaluation...")
        best_model_state = torch.load('Best_Fhe_friendly_qat_ensemble_model.pth', map_location=DEVICE)
        qat_ensemble.load_state_dict(best_model_state)
        qat_ensemble.to(DEVICE)  # Ensure the model is on the correct device after loading

        test_accuracy = evaluate_ensemble(qat_ensemble, test_loader, DEVICE)
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")

        print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()