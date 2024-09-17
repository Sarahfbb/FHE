import numpy as np
import os
import time
from tqdm import tqdm
from concrete.ml.deployment import FHEModelClient, FHEModelServer

FHE_DIRECTORY = './fhe_client_server_files/'
KEY_DIRECTORY = './keys_client/'

def main():
    start_time = time.time()

    # Load test features and labels
    print("Loading test features and labels...")
    test_features = np.load('features/1TestF.npy')
    test_labels = np.load('features/1TestT.npy')
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
        # Quantize, encrypt, and serialize
        encrypted_input = client.quantize_encrypt_serialize(feature.reshape(1, -1))

        # Measure execution time for server processing
        time_begin = time.time()
        encrypted_result = server.run(encrypted_input, serialized_evaluation_keys)
        time_end = time.time()
        execution_times.append(time_end - time_begin)

        # Client decrypts the result
        decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
        decrypted_predictions.append(decrypted_prediction)

    # Print performance metrics
    print(f"Encrypted data is {len(encrypted_input)/feature.nbytes:.2f} times larger than the clear data")
    print(f"The average execution time is {np.mean(execution_times):.2f} seconds per sample.")

    # Calculate accuracy
    print("Calculating accuracy...")
    predictions = np.array(decrypted_predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == test_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()