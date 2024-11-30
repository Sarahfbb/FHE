import numpy as np
import torch
import os
import time
from tqdm import tqdm
from concrete.ml.deployment import FHEModelClient, FHEModelServer
from sklearn.preprocessing import StandardScaler
import joblib

FHE_DIRECTORY = './fhe_client_server_files/'
KEY_DIRECTORY = './keys_client/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(features_path, labels_path, scaler_path=None):
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        features = scaler.transform(features)
    else:
        print("Warning: No scaler found. Using raw features.")
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    
    return features_tensor, labels_tensor

def main():
    print(f"Using device: {DEVICE}")
    start_time = time.time()
    try:
        # Load test features and labels
        print("Loading test features and labels...")
        test_features, test_labels = load_and_preprocess_data(
            'features/Test_Features.npy',
            'features/Test_Targets.npy',
            'features/scaler.joblib'
        )
        print(f"Number of test samples: {len(test_features)}")

        # Load the existing client and server
        print("Loading existing client and server...")
        client = FHEModelClient(path_dir=FHE_DIRECTORY, key_dir=KEY_DIRECTORY)
        server = FHEModelServer(path_dir=FHE_DIRECTORY)

        # Get evaluation keys (this should be done only once)
        serialized_evaluation_keys = client.get_serialized_evaluation_keys()

        # Process test set
        print("Processing test set...")
        decrypted_predictions = []
        execution_times = []

        for feature in tqdm(test_features, desc="Processing test set"):
            # Move to CPU for FHE operations
            feature_cpu = feature.cpu().numpy()
            
            # Quantize, encrypt, and serialize
            encrypted_input = client.quantize_encrypt_serialize(feature_cpu.reshape(1, -1))
            
            # Measure execution time for server processing
            time_begin = time.time()
            encrypted_result = server.run(encrypted_input, serialized_evaluation_keys)
            time_end = time.time()
            execution_times.append(time_end - time_begin)
            
            # Client decrypts the result
            decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
            decrypted_predictions.append(decrypted_prediction)

        # Print performance metrics
        print(f"Encrypted data is {len(encrypted_input)/feature_cpu.nbytes:.2f} times larger than the clear data")
        print(f"The average execution time is {np.mean(execution_times):.2f} seconds per sample.")

        # Calculate accuracy
        print("Calculating accuracy...")
        predictions = torch.tensor(decrypted_predictions, device=DEVICE)
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = (predicted_labels == test_labels).float().mean().item()
        print(f"Test accuracy: {accuracy:.4f}")

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred during inference: {str(e)}")

if __name__ == "__main__":
    main()