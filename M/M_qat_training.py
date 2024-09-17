import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import os
import psutil

# Brevitas imports
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.nn import QuantIdentity, QuantLinear

print("Current working directory:", os.getcwd())

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-7
DEVICE = torch.device("cpu")
BIT_WIDTH = 6
NUM_CLASSES = 10  # Number of classes in the dataset

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class QuantBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, bit_width=BIT_WIDTH):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)
        self.quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        x = self.bn(x.value)
        return self.quant(x)
    
class QuantFHEFriendlyMLPClassifier(torch.nn.Module):
    def __init__(
        self, 
        input_dim, 
        num_classes, 
        bit: int = BIT_WIDTH,
        act_quant: torch.nn.Module = Int8ActPerTensorFloat,
        weight_quant: torch.nn.Module = Int8WeightPerTensorFloat,
    ):
        super(QuantFHEFriendlyMLPClassifier, self).__init__()
        
        self.input_quant = QuantIdentity(
            bit_width=bit, act_quant=act_quant, return_quant_tensor=True
        )
        
        self.fc1 = QuantLinear(
            input_dim, 
            256, 
            bias=False,
            weight_bit_width=bit,
            weight_quant=weight_quant,
            return_quant_tensor=True
        )
        self.bn1 = QuantBatchNorm1d(256, bit_width=bit)
        self.act1 = QuantIdentity(return_quant_tensor=True, bit_width=bit, act_quant=act_quant)
        
        self.fc2 = QuantLinear(
            256,
            128,
            bias=False,
            weight_bit_width=bit,
            weight_quant=weight_quant,
            return_quant_tensor=True
        )
        self.bn2 = QuantBatchNorm1d(128, bit_width=bit)
        self.act2 = QuantIdentity(return_quant_tensor=True, bit_width=bit, act_quant=act_quant)
        
        self.fc3 = QuantLinear(
            128,
            num_classes, 
            bias=True,
            weight_bit_width=bit,
            weight_quant=weight_quant,
            return_quant_tensor=True
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(self.poly_activation(x))
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(self.poly_activation(x))
        x = self.fc3(x)
        return x.value

    @staticmethod
    def poly_activation(x):
        return 0.5 * x + 0.25 * x * x

class EnsembleQATModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EnsembleQATModel, self).__init__()
        self.models = nn.ModuleList([QuantFHEFriendlyMLPClassifier(input_dim, 1) for _ in range(NUM_CLASSES)])
        self.num_classes = num_classes

    def forward(self, x):
        outputs = torch.cat([model(x) for model in self.models], dim=1)
        return outputs

    def load_individual_models(self, device):
        best_ensemble = torch.load('best_ensembles.pt', map_location=device)
        for class_id, model_num in enumerate(best_ensemble):
            model_filename = f'fhe_friendly_ensemble_class_{class_id}.pt'
            if os.path.exists(model_filename):
                class_models = torch.load(model_filename, map_location=device)
                self.models[class_id].load_state_dict(class_models[model_num], strict=False)
                print(f"Loaded weights for class {class_id}, model {model_num}")
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

def train_ensemble_qat(model, train_loader, val_loader, param, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    with tqdm(total=param['epochs'], file=sys.stdout) as pbar:
        for epoch in range(param['epochs']):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
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

            scheduler.step()

            train_accuracy = 100 * correct / total
            val_accuracy = evaluate_ensemble(model, val_loader, device)

            param['accuracy_train'].append(train_accuracy)
            param['accuracy_test'].append(val_accuracy)
            param['loss_train_history'].append(running_loss / len(train_loader))

            pbar.set_description(f"Epoch {epoch+1}/{param['epochs']}")
            pbar.set_postfix({
                'Train Loss': f"{running_loss/len(train_loader):.4f}",
                'Train Acc': f"{train_accuracy:.2f}%",
                'Val Acc': f"{val_accuracy:.2f}%"
            })
            pbar.update(1)

    torch.save(model.state_dict(), 'best_qat_ensembles.pt')
    return model

def plot_baseline(param: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    epochs = range(1, len(param["accuracy_test"]) + 1)
    
    ax1.plot(epochs, param["accuracy_test"], marker="o", label="Test accuracy")
    ax1.plot(epochs, param["accuracy_train"], marker="o", label="Train accuracy")
    
    baseline = 74.2  # Set the baseline to 74.2%
    ax1.axhline(y=baseline, color='r', linestyle='--', label="Baseline")
    ax1.text(1, baseline + 0.5, f"Baseline = {baseline:.1f}%", fontsize=12, color="red")

    min_acc = min(min(param["accuracy_test"]), min(param["accuracy_train"]), baseline)
    max_acc = max(max(param["accuracy_test"]), max(param["accuracy_train"]), baseline)
    y_range = max_acc - min_acc
    ax1.set_ylim([max(0, min_acc - 0.1 * y_range), min(100, max_acc + 0.1 * y_range)])

    ax1.set_title("Accuracy during Quantization")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Epochs")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2.plot(epochs, param["loss_train_history"], marker="o", label="Train Loss")
    ax2.set_title("Training Loss during Quantization")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")
    ax2.grid(True)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig('ensemble_qat_plot.png')
    plt.close()

class MemoryEfficientDataset(torch.utils.data.Dataset):
    def __init__(self, features_file, targets_file, scaler=None):
        self.features = np.load(features_file, mmap_mode='r')
        self.targets = np.load(targets_file)
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.fit_scaler()

    def fit_scaler(self):
        batch_size = 1000
        for i in range(0, len(self.features), batch_size):
            batch = self.features[i:i+batch_size]
            self.scaler.partial_fit(batch)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = self.scaler.transform([feature])[0]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)

def main():
    try:
        torch.cuda.empty_cache()
        
        print("Loading data...")
        print_memory_usage()

        # Create datasets
        train_dataset = MemoryEfficientDataset('1TF.npy', '1TT.npy')
        val_dataset = MemoryEfficientDataset('1VF.npy', '1VT.npy', train_dataset.scaler)
        test_dataset = MemoryEfficientDataset('1TestF.npy', '1TestT.npy', train_dataset.scaler)

        print(f"Train features shape: {train_dataset.features.shape}")
        print(f"Train targets shape: {train_dataset.targets.shape}")
        print(f"Val features shape: {val_dataset.features.shape}")
        print(f"Test features shape: {test_dataset.features.shape}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        print("DataLoaders created.")
        print_memory_usage()

        input_dim, num_classes = train_dataset.features.shape[1], len(np.unique(train_dataset.targets))
        ensemble_qat_model = EnsembleQATModel(input_dim, num_classes).to(DEVICE)

        # Load individual class models
        print("Loading individual class models...")
        ensemble_qat_model.load_individual_models(DEVICE)
        print("Individual class models loaded.")

        param = {
            'lr': LEARNING_RATE,
            'epochs': EPOCHS,
            'accuracy_train': [],
            'accuracy_test': [],
            'loss_train_history': [],
        }

        print("Training Ensemble QAT Model...")
        train_ensemble_qat(ensemble_qat_model, train_loader, val_loader, param, DEVICE)

        ensemble_qat_model.load_state_dict(torch.load('best_qat_ensembles.pt', map_location=DEVICE))

        test_accuracy = evaluate_ensemble(ensemble_qat_model, test_loader, DEVICE)
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")

        param['baseline_accuracy'] = test_accuracy  # Use the final test accuracy as the baseline

        plot_baseline(param)
        print("Plots saved as 'ensemble_qat_plot.png'")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()