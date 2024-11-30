import numpy as np
import os
import time
import torch
from tqdm import tqdm
from S_qat_training import QuantFHEFriendlyMLPClassifier

FHE_DISABLED_DIRECTORY = './fhe_disabled_files/'

def test_fhe_disabled(test_features, test_labels, input_dim, num_classes):
    print("Testing FHE-disabled model...")
    model = QuantFHEFriendlyMLPClassifier(input_dim, num_classes)
    model.load_state_dict(torch.load(os.path.join(FHE_DISABLED_DIRECTORY, 'fhe_disabled_model.pth')))
    model.eval()

    predictions = []
    execution_times = []

    with torch.no_grad():
        for feature in tqdm(test_features, desc="Processing test set (FHE-disabled)"):
            time_begin = time.time()
            output = model(torch.tensor(feature.reshape(1, -1), dtype=torch.float32))
            time_end = time.time()
            execution_times.append(time_end - time_begin)
            predictions.append(output.detach().numpy())

    predictions = np.concatenate(predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of predicted_labels: {predicted_labels.shape}")
    print(f"Shape of test_labels: {test_labels.shape}")
    
    print(f"First few predictions: {predictions[:5]}")
    print(f"First few predicted labels: {predicted_labels[:5]}")
    print(f"First few true labels: {test_labels[:5]}")

    accuracy = np.mean(predicted_labels == test_labels)
    avg_execution_time = np.mean(execution_times)

    return accuracy, avg_execution_time

def main():
    start_time = time.time()

    print("Loading test features and labels...")
    test_features = np.load('features/1TestF.npy')
    test_labels = np.load('features/1TestT.npy')
    input_dim, num_classes = test_features.shape[1], len(np.unique(test_labels))
    print(f"Number of test samples: {len(test_features)}")
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")

    fhe_disabled_accuracy, fhe_disabled_time = test_fhe_disabled(test_features, test_labels, input_dim, num_classes)

    print("\nResults:")
    print(f"FHE-disabled accuracy: {fhe_disabled_accuracy:.4f}")
    print(f"FHE-disabled average execution time: {fhe_disabled_time:.6f} seconds per sample")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()