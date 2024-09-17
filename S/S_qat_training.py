import time
import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os

# Brevitas imports
import brevitas
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from brevitas.nn import QuantIdentity, QuantLinear
from concrete.ml.torch.compile import compile_brevitas_qat_model

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-7
DEVICE = torch.device("cpu")
BIT_WIDTH = 6

class QuantBatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, bit_width=BIT_WIDTH):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features)
        self.quant = QuantIdentity(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        x = self.bn(x.value)  # Use .value if x is a QuantTensor
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
            bias=False,  # Set to False when using BatchNorm
            weight_bit_width=bit,
            weight_quant=weight_quant,
            return_quant_tensor=True
        )
        self.bn1 = QuantBatchNorm1d(256, bit_width=bit)
        self.act1 = QuantIdentity(return_quant_tensor=True, bit_width=bit, act_quant=act_quant)
        
        self.fc2 = QuantLinear(
            256,
            128,
            bias=False,  # Set to False when using BatchNorm
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

def evaluate_mlp(model, data_loader, device):
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
            val_accuracy = evaluate_mlp(model, val_loader, device)

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

    # Save the final model
    torch.save(model.state_dict(), 'Good_Fhe_friendly_qat_mlp_model.pth')
    return model

def main():
    try:
        torch.cuda.empty_cache()
        
        # Load and preprocess data
        print("Loading data...")
        features_dir = 'features'
        train_features = numpy.load(os.path.join(features_dir, '1TF.npy'))
        train_targets = numpy.load(os.path.join(features_dir, '1TT.npy'))
        val_features = numpy.load(os.path.join(features_dir, '1VF.npy'))
        val_targets = numpy.load(os.path.join(features_dir, '1VT.npy'))
        test_features = numpy.load(os.path.join(features_dir, '1TestF.npy'))
        test_targets = numpy.load(os.path.join(features_dir, '1TestT.npy'))
        print("Data loaded successfully.")

        print("Preprocessing data...")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        print("Data preprocessing completed.")

        train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32, device=DEVICE), 
                              torch.tensor(train_targets, dtype=torch.long, device=DEVICE))
        val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32, device=DEVICE), 
                            torch.tensor(val_targets, dtype=torch.long, device=DEVICE))
        test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32, device=DEVICE), 
                             torch.tensor(test_targets, dtype=torch.long, device=DEVICE))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        print("DataLoaders created.")

        input_dim, num_classes = train_features.shape[1], len(numpy.unique(train_targets))
        qat_mlp = QuantFHEFriendlyMLPClassifier(input_dim, num_classes).to(DEVICE)

        if os.path.exists('Good_Fhe_friendly_mlp_model.pth'):
            print("Loading pre-trained weights...")
            pre_trained_weights = torch.load('Good_Fhe_friendly_mlp_model.pth', map_location=DEVICE)
            qat_mlp.load_state_dict(pre_trained_weights, strict=False)
            print("Pre-trained weights loaded and mapped to the quantized model.")
        else:
            print("No pre-trained weights found. Starting from scratch.")

        param = {
            'lr': LEARNING_RATE,
            'epochs': EPOCHS,
            'accuracy_train': [],
            'accuracy_test': [],
            'loss_train_history': [],
        }

        print("Training Quantized MLP...")
        train_qat(qat_mlp, train_loader, val_loader, param, DEVICE)

        qat_mlp.load_state_dict(torch.load('Good_Fhe_friendly_qat_mlp_model.pth', map_location=DEVICE))

        test_accuracy = evaluate_mlp(qat_mlp, test_loader, DEVICE)
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()