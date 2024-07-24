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
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Start timing
start_time = time.time()

# Set up device
device = torch.device("cpu")
print(f"Using device: {device}")

class BrevitasFHEFriendlyMLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BrevitasFHEFriendlyMLPClassifier, self).__init__()
        self.input_quant = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(input_dim, 256, bias=True, weight_bit_width=8, 
                                   bias_quant=BiasQuant, weight_quant=WeightQuant)
        self.act1 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(256, 128, bias=True, weight_bit_width=8, 
                                   bias_quant=BiasQuant, weight_quant=WeightQuant)
        self.act2 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(128, 1, bias=True, weight_bit_width=8, 
                                   bias_quant=BiasQuant, weight_quant=WeightQuant)

    def forward(self, x):
        x = self.input_quant(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

def copy_weights_from_pytorch_to_brevitas(pytorch_model, brevitas_model):
    brevitas_model.fc1.weight.data.copy_(pytorch_model['fc1.weight'])
    brevitas_model.fc1.bias.data.copy_(pytorch_model['fc1.bias'])
    brevitas_model.fc2.weight.data.copy_(pytorch_model['fc2.weight'])
    brevitas_model.fc2.bias.data.copy_(pytorch_model['fc2.bias'])
    brevitas_model.fc3.weight.data.copy_(pytorch_model['fc3.weight'])
    brevitas_model.fc3.bias.data.copy_(pytorch_model['fc3.bias'])

def fine_tune_quantized(model, train_loader, criterion, optimizer, scheduler, epochs, class_id):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Class {class_id}, Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.8f}')
        scheduler.step(avg_loss)

    return model

def evaluate_ensemble(ensemble, features, targets):
    num_classes = len(ensemble)
    num_samples = len(features)
    predictions = torch.zeros(num_samples, num_classes, device=device)
    
    for class_id, model in enumerate(ensemble):
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            predictions[:, class_id] = outputs.squeeze()
    
    # Convert logits to probabilities
    probabilities = torch.sigmoid(predictions)
    
    # Get the class with the highest probability for each sample
    ensemble_pred = torch.argmax(probabilities, dim=1)
    
    accuracy = (ensemble_pred == targets).float().mean().item()
    return accuracy * 100

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

    # Load the best ensemble
    best_ensemble = torch.load('best_ensembles.pt')
    
    input_dim = val_features.shape[1]

    quantized_ensemble = []

    for class_id, model_num in enumerate(best_ensemble):
        print(f"Fine-tuning and quantizing model for class {class_id}")

        # Load the pre-trained PyTorch model state dict
        pytorch_state_dict = torch.load(f'fhe_friendly_ensemble_class_{class_id}_mlp_{model_num}.pt')
        
        # Initialize the Brevitas model
        brevitas_model = BrevitasFHEFriendlyMLPClassifier(input_dim).to(device)
        
        # Copy weights from PyTorch model to Brevitas model
        copy_weights_from_pytorch_to_brevitas(pytorch_state_dict, brevitas_model)

        # Create binary targets for this class
        binary_targets = (val_targets == class_id).astype(float)

        # Create DataLoader for validation (to be used as training data for fine-tuning)
        val_dataset = TensorDataset(val_features_tensor, torch.tensor(binary_targets, dtype=torch.float32).to(device))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Define optimizer and loss function for fine-tuning
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(brevitas_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # Fine-tune the quantized model
        print(f"Fine-tuning quantized MLP for class {class_id}...")
        fine_tuned_quantized_model = fine_tune_quantized(brevitas_model, val_loader, criterion, optimizer, scheduler, 
                                                         epochs=FINE_TUNING_EPOCHS, class_id=class_id)

        quantized_ensemble.append(fine_tuned_quantized_model)
        torch.save(fine_tuned_quantized_model.state_dict(), f'Mfine_tuned_quantized_model_class_{class_id}.pth')

    # Evaluate the final ensemble on validation and test data
    val_accuracy = evaluate_ensemble(quantized_ensemble, val_features_tensor, val_targets_tensor)
    test_accuracy = evaluate_ensemble(quantized_ensemble, test_features_tensor, test_targets_tensor)
    print(f"Final ensemble accuracies - Val: {val_accuracy:.2f}%, Test: {test_accuracy:.2f}%")

    # Save the final quantized ensemble
    torch.save([model.state_dict() for model in quantized_ensemble], 'Mfinal_quantized_ensemble.pt')

    end_time = time.time()
    print(f"\nTotal running time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()