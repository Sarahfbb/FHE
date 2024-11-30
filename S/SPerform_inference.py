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
BATCH_SIZE = 128  # Changed to 128 as requested

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

def save_predictions(predictions, filename='decrypted_predictions.npy'):
    np.save(filename, np.array(predictions))
    print(f"Predictions saved to {filename}")

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

        for i in tqdm(range(0, len(test_features), BATCH_SIZE), desc="Processing batches"):
            batch = test_features[i:i+BATCH_SIZE]
            batch_cpu = batch.cpu().numpy()

            # Measure execution time for batch processing
            time_begin = time.time()

            # Quantize, encrypt, and serialize
            encrypted_input = client.quantize_encrypt_serialize(batch_cpu)

            # Run the server
            encrypted_result = server.run(encrypted_input, serialized_evaluation_keys)

            # Decrypt the result
            decrypted_batch = client.deserialize_decrypt_dequantize(encrypted_result)

            time_end = time.time()
            execution_times.append(time_end - time_begin)
            decrypted_predictions.extend(decrypted_batch)

        # Print performance metrics
        print(f"The average execution time is {np.mean(execution_times):.2f} seconds per batch.")
        print(f"The total execution time for inference is {sum(execution_times):.2f} seconds.")

        # Calculate accuracy
        print("Calculating accuracy...")
        predictions = torch.tensor(decrypted_predictions, device=DEVICE)
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = (predicted_labels == test_labels).float().mean().item()
        print(f"Test accuracy: {accuracy:.4f}")

        # Save predictions
        save_predictions(decrypted_predictions)

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()