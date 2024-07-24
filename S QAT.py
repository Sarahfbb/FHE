
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant
from brevitas.quant import Int8ActPerTensorFloat as ActQuant
from brevitas.quant import Int8WeightPerTensorFloat as WeightQuant

# Configuration
FINE_TUNING_EPOCHS = 10
BATCH_SIZE = 64

# Start timing
start_time = time.time()

# Set up device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

class BrevitasFHEFriendlyMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BrevitasFHEFriendlyMLPClassifier, self).__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(input_dim, 256, bias=True, weight_bit_width=8, 
                                   bias_quant=BiasQuant, weight_quant=WeightQuant)
        self.act1 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(256, 128, bias=True, weight_bit_width=8, 
                                   bias_quant=BiasQuant, weight_quant=WeightQuant)
        self.act2 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(128, num_classes, bias=True, weight_bit_width=8, 
                                   bias_quant=BiasQuant, weight_quant=WeightQuant)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

def copy_weights_from_pytorch_to_brevitas(pytorch_model, brevitas_model):
    brevitas_model.fc1.weight.data.copy_(pytorch_model['fc1.weight'])
    brevitas_model.fc1.bias.data.copy_(pytorch_model['fc1.bias'])
    brevitas_model.fc2.weight.data.copy_(pytorch_model['fc2.weight'])
    brevitas_model.fc2.bias.data.copy_(pytorch_model['fc2.bias'])
    brevitas_model.fc3.weight.data.copy_(pytorch_model['fc3.weight'])
    brevitas_model.fc3.bias.data.copy_(pytorch_model['fc3.bias'])

def evaluate_mlp(model, features, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100. * correct / len(labels)
    return accuracy

def fine_tune_quantized(model, train_loader, val_features, val_labels, criterion, optimizer, scheduler, epochs):
    best_val_accuracy = 0
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_accuracy = evaluate_mlp(model, val_features, val_labels)
        avg_loss = running_loss / len(train_loader)
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.8f}, '
              f'Val Accuracy: {val_accuracy:.2f}%, Time: {time.time() - epoch_start:.2f} seconds')
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), ' Sfine_tuned_quantized_mlp_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return model

def main():
    # Load pre-extracted features
    val_features = np.load('val_features.npy')
    val_targets = np.load('val_targets.npy')
    test_features = np.load('test_features.npy')
    test_targets = np.load('test_targets.npy')

    # Standardize features
    scaler = StandardScaler()
    val_features = scaler.fit_transform(val_features)
    test_features = scaler.transform(test_features)

    # Convert to PyTorch tensors
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.long).to(device)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.long).to(device)

    # Create DataLoader for validation (to be used as training data for fine-tuning)
    val_dataset = TensorDataset(val_features_tensor, val_targets_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load the pre-trained PyTorch model state dict
    input_dim = val_features.shape[1]
    num_classes = 10
    pytorch_state_dict = torch.load('fhe_friendly_mlp_model.pth')
    
    # Initialize the Brevitas model
    brevitas_model = BrevitasFHEFriendlyMLPClassifier(input_dim, num_classes).to(device)
    
    # Copy weights from PyTorch model to Brevitas model
    copy_weights_from_pytorch_to_brevitas(pytorch_state_dict, brevitas_model)

    # Define optimizer and loss function for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(brevitas_model.parameters(), lr=1e-4, weight_decay=1e-5)                   
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # Fine-tune the quantized model
    print("Fine-tuning quantized MLP...")
    fine_tuned_quantized_model = fine_tune_quantized(brevitas_model, val_loader, val_features_tensor, val_targets_tensor, 
                                                     criterion, optimizer, scheduler, epochs=FINE_TUNING_EPOCHS)

    # Final evaluation
    val_accuracy = evaluate_mlp(fine_tuned_quantized_model, val_features_tensor, val_targets_tensor)
    test_accuracy = evaluate_mlp(fine_tuned_quantized_model, test_features_tensor, test_targets_tensor)
    print(f"Final Accuracies - Val: {val_accuracy:.2f}%, Test: {test_accuracy:.2f}%")

    # Save the final quantized model
    torch.save(fine_tuned_quantized_model.state_dict(), 'Sfinal_brevitas_quantized_mlp_model.pth')

    end_time = time.time()
    print(f"\nTotal running time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()