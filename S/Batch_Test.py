import numpy as np
import os
import time
import argparse
from tqdm import tqdm
from concrete.ml.deployment import FHEModelClient, FHEModelServer

FHE_DIRECTORY = './fhe_client_server_files/'
KEY_DIRECTORY = './keys_client/'

def process_range(start_idx, end_idx, test_features, test_labels, client, server):
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    decrypted_predictions = []
    execution_times = []
    for feature in tqdm(test_features[start_idx:end_idx], desc=f"Processing samples {start_idx} to {end_idx}"):
        encrypted_input = client.quantize_encrypt_serialize(feature.reshape(1, -1))
        time_begin = time.time()
        encrypted_result = server.run(encrypted_input, serialized_evaluation_keys)
        time_end = time.time()
        execution_times.append(time_end - time_begin)
        decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
        decrypted_predictions.append(decrypted_prediction)
    print(f"Encrypted data is {len(encrypted_input)/feature.nbytes:.2f} times larger than the clear data")
    print(f"The average execution time is {np.mean(execution_times):.2f} seconds per sample.")
    predictions = np.array(decrypted_predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == test_labels[start_idx:end_idx])
    print(f"Test accuracy for samples {start_idx} to {end_idx}: {accuracy:.4f}")
    return predictions, execution_times

def combine_results(total_samples):
    all_predictions = []
    all_execution_times = []
    for start in range(0, total_samples, 50):  # Assuming batches of 50
        end = min(start + 50, total_samples)
        pred_file = f'predictions_range_{start}_{end}.npy'
        time_file = f'execution_times_range_{start}_{end}.npy'
        if os.path.exists(pred_file) and os.path.exists(time_file):
            all_predictions.append(np.load(pred_file))
            all_execution_times.extend(np.load(time_file))
    
    if all_predictions:
        all_predictions = np.vstack(all_predictions)
        test_labels = np.load('features/1TestT.npy')
        predicted_labels = np.argmax(all_predictions, axis=1)
        overall_accuracy = np.mean(predicted_labels == test_labels[:len(predicted_labels)])
        avg_execution_time = np.mean(all_execution_times)
        
        print(f"Overall test accuracy: {overall_accuracy:.4f}")
        print(f"Average execution time: {avg_execution_time:.2f} seconds per sample")
        print(f"Total samples processed: {len(predicted_labels)}")
    else:
        print("No results found to combine.")

def main():
    parser = argparse.ArgumentParser(description='Process FHE test samples in a specified range or combine results.')
    parser.add_argument('--start', type=int, help='Start index of the range to process')
    parser.add_argument('--end', type=int, help='End index of the range to process')
    parser.add_argument('--combine', action='store_true', help='Combine results from all processed ranges')
    args = parser.parse_args()

    if args.combine:
        print("Combining results from all processed ranges...")
        test_features = np.load('features/1TestF.npy')
        combine_results(len(test_features))
    elif args.start is not None and args.end is not None:
        start_time = time.time()
        print("Loading test features and labels...")
        test_features = np.load('features/1TestF.npy')
        test_labels = np.load('features/1TestT.npy')
        print(f"Number of test samples: {len(test_features)}")
        print("Loading existing client and server...")
        client = FHEModelClient(path_dir=FHE_DIRECTORY, key_dir=KEY_DIRECTORY)
        server = FHEModelServer(path_dir=FHE_DIRECTORY)
        predictions, execution_times = process_range(args.start, args.end, test_features, test_labels, client, server)
        # Save results
        np.save(f'predictions_range_{args.start}_{args.end}.npy', predictions)
        np.save(f'execution_times_range_{args.start}_{args.end}.npy', execution_times)
        total_time = time.time() - start_time
        print(f"\nTotal execution time for range {args.start} to {args.end}: {total_time:.2f} seconds")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

    # python /home/sarah/Desktop/FHE/S/Batch_Test.py --start 0 --end 50 &> result_50-100.out