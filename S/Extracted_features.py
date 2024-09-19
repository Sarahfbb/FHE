import os
import time
import numpy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

start_time = time.time()

# Set the device to CUDA:0
device = torch.device("cuda:0")
print(f"Using device: {device}")

# Load the VGG16 model and move it to CUDA:0
try:
    vgg_model = torchvision.models.vgg16(weights='IMAGENET1K_V1').to(device)
except TypeError:
    vgg_model = torchvision.models.vgg16(pretrained=True).to(device)

# Define a custom FHE-friendly activation function
class PolynomialActivation(nn.Module):
    def forward(self, x):
        return 0.5 * x + 0.25 * x.pow(2)

# Define a custom VGG feature extractor class with reduced dimension and FHE-friendly components
class FHEFriendlyVGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_model, reduced_dim=100):
        super(FHEFriendlyVGGFeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(vgg_model.features.children())[:-1])  
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_reducer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            PolynomialActivation(),
            nn.Linear(4096, reduced_dim),
            PolynomialActivation()
        )
        self.classifier = nn.Linear(reduced_dim, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.feature_reducer(x)
        output = self.classifier(features)
        return features, output

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Reduce the number of samples for training set
num_train_samples = 1000
train_indices = torch.randperm(len(cifar10_train))[:num_train_samples]
reduced_cifar10_train = Subset(cifar10_train, train_indices)

# Reduce the number of samples for test set
num_test_samples = 100
test_indices = torch.randperm(len(cifar10_test))[:num_test_samples]
reduced_cifar10_test = Subset(cifar10_test, test_indices)

# Split the reduced training dataset
train_ratio, val_ratio = 0.7, 0.3
train_size = int(train_ratio * num_train_samples)
val_size = num_train_samples - train_size

train_dataset, val_dataset = random_split(reduced_cifar10_train, [train_size, val_size])

# Create data loaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(reduced_cifar10_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the VGG feature extractor and move it to CUDA:0
reduced_dim = 500
vgg_feature_extractor = FHEFriendlyVGGFeatureExtractor(vgg_model, reduced_dim=reduced_dim).to(device)

# Fine-tuning function
def fine_tune_vgg(model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {100.*correct/total:.2f}%')
    
    return model

# Fine-tune the VGG feature extractor
print("Fine-tuning the VGG feature extractor...")
vgg_feature_extractor = fine_tune_vgg(vgg_feature_extractor, train_loader, val_loader)

def extract_features_and_targets(model, loader):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Extracting features and targets"):
            data = data.to(device)
            feature, _ = model(data)
            features.append(feature.cpu().numpy())
            targets.append(target.numpy())
    
    features = numpy.concatenate(features, axis=0)
    targets = numpy.concatenate(targets, axis=0)
    return features, targets

# Extract features and targets using the fine-tuned VGG model
print("Extracting features and targets for training set...")
train_features, train_targets = extract_features_and_targets(vgg_feature_extractor, train_loader)

print("Extracting features and targets for validation set...")
val_features, val_targets = extract_features_and_targets(vgg_feature_extractor, val_loader)

print("Extracting features and targets for test set...")
test_features, test_targets = extract_features_and_targets(vgg_feature_extractor, test_loader)

# Print shapes of extracted features and targets
print(f"Train features shape: {train_features.shape}, Train targets shape: {train_targets.shape}")
print(f"Validation features shape: {val_features.shape}, Validation targets shape: {val_targets.shape}")
print(f"Test features shape: {test_features.shape}, Test targets shape: {test_targets.shape}")

# Create a 'features' directory if it doesn't exist
features_dir = 'features'
os.makedirs(features_dir, exist_ok=True)

# Save features and targets in the 'features' directory
numpy.save(os.path.join(features_dir, '1TF.npy'), train_features)
numpy.save(os.path.join(features_dir, '1TT.npy'), train_targets)
numpy.save(os.path.join(features_dir, '1VF.npy'), val_features)
numpy.save(os.path.join(features_dir, '1VT.npy'), val_targets)
numpy.save(os.path.join(features_dir, '1TestF.npy'), test_features)
numpy.save(os.path.join(features_dir, '1TestT.npy'), test_targets)

print(f"Features and targets saved in the '{features_dir}' directory.")
print(f"Script finished in {time.time() - start_time} seconds.")
