import os
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin

start_time = time.time()

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Try to load the VGG16 model using the appropriate API based on the torchvision version
try:
    vgg_model = torchvision.models.vgg16(weights='IMAGENET1K_V1').to(device)
except TypeError:
    vgg_model = torchvision.models.vgg16(pretrained=True).to(device)

# Define a custom VGG feature extractor class
class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_model):
        super(VGGFeatureExtractor, self).__init__()
        self.features = vgg_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(512 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        output = self.classifier(features)
        return features, output

# Improved data augmentation
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Initialize the VGG feature extractor
vgg_feature_extractor = VGGFeatureExtractor(vgg_model).to(device)

# Load CIFAR-10 dataset
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split the dataset into train, validation, and test sets
train_ratio = 0.5
val_ratio = 0.2
test_ratio = 0.3
train_size = int(train_ratio * len(cifar10_train))
val_size = int(val_ratio * len(cifar10_train))
test_size = len(cifar10_train) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(cifar10_train, [train_size, val_size, test_size])

# Define batch size
batch_size = 256

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Fine-tuning function
def fine_tune_vgg(model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
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
            for inputs, labels in val_loader:
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
vgg_feature_extractor = fine_tune_vgg(vgg_feature_extractor, train_loader, val_loader)

# Function to extract features using the VGG model
def extract_features(model, loader):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            x_feat, _ = model(data)  # Extract features from the model
            features.append(x_feat.cpu().numpy())
            targets.append(target.cpu().numpy())
    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    return features, targets

# Extract features using the fine-tuned VGG model
train_features, train_targets = extract_features(vgg_feature_extractor, train_loader)
val_features, val_targets = extract_features(vgg_feature_extractor, val_loader)
test_features, test_targets = extract_features(vgg_feature_extractor, test_loader)

# Save extracted features
np.save('train_features.npy', train_features)
np.save('train_targets.npy', train_targets)
np.save('val_features.npy', val_features)
np.save('val_targets.npy', val_targets)
np.save('test_features.npy', test_features)
np.save('test_targets.npy', test_targets)

print(f"Script finished in {time.time() - start_time} seconds.")
