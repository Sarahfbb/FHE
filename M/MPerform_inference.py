import numpy as np
import torch
import os
import time
from tqdm import tqdm
from concrete.ml.deployment import FHEModelClient, FHEModelServer
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

FHE_DIRECTORY = './fhe_ensemble_client_server_files/'
KEY_DIRECTORY = './keys_client_ensemble/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 20
NUM_CLASSES = 10

def get_chunk_info(total_samples, chunk_index):
    """Get information for a specific chunk"""
    start_idx = chunk_index * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, total_samples)
    
    if start_idx >= total_samples:
        raise ValueError(f"Chunk index {chunk_index} is out of range. Total samples: {total_samples}")
    
    return {
        'chunk_number': chunk_index,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'size': end_idx - start_idx
    }


def load_and_preprocess_data(features_path, labels_path, scaler_path=None):
    """Load and preprocess the test data"""
    print(f"Loading data from {features_path} and {labels_path}")
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    if scaler_path and os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        features = scaler.transform(features)
    else:
        print("Warning: No scaler found. Using raw features.")
    
    features_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    return features_tensor, labels_tensor

def process_chunk(chunk_info, features, labels, clients, servers):
    """Process a specific chunk of data"""
    print(f"\nProcessing Chunk {chunk_info['chunk_number']:03d}")
    print(f"Range: {chunk_info['start_idx']} to {chunk_info['end_idx']} (Size: {chunk_info['size']})")
    
    # Get the chunk data
    chunk_features = features[chunk_info['start_idx']:chunk_info['end_idx']]
    
    chunk_predictions = []
    chunk_times = []
    
    # Add intermediate saves every 5 samples
    save_interval = 5
    last_save = 0
    
    try:
        progress_bar = tqdm(
            range(len(chunk_features)),
            desc=f"Chunk {chunk_info['chunk_number']:03d}",
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for i in progress_bar:
            sample = chunk_features[i:i+1].cpu().numpy()
            time_begin = time.time()
            
            # Process each binary classifier
            sample_results = []
            for j in range(NUM_CLASSES):
                encrypted_input = clients[j].quantize_encrypt_serialize(sample)
                serialized_evaluation_keys = clients[j].get_serialized_evaluation_keys()
                encrypted_result = servers[j].run(encrypted_input, serialized_evaluation_keys)
                decrypted_result = clients[j].deserialize_decrypt_dequantize(encrypted_result)
                sample_results.append(decrypted_result)
            
            combined_result = np.array(sample_results).T
            chunk_predictions.append(combined_result[0])
            
            time_end = time.time()
            chunk_times.append(time_end - time_begin)
            
            # Save intermediate results every save_interval samples
            if (i + 1) % save_interval == 0:
                temp_predictions = np.array(chunk_predictions)
                temp_times = np.array(chunk_times)
                np.save(f'chunk_results/temp_chunk{chunk_info["chunk_number"]:03d}_predictions.npy', temp_predictions)
                np.save(f'chunk_results/temp_chunk{chunk_info["chunk_number"]:03d}_times.npy', temp_times)
                last_save = i + 1
            
            if (i + 1) % 10 == 0:
                avg_time = np.mean(chunk_times[-10:])
                progress_bar.set_postfix({'Avg time/sample': f'{avg_time:.2f}s'})
        
        # Save final results without accuracy calculation
        chunk_predictions_array = np.array(chunk_predictions)
        save_chunk_results(chunk_info, chunk_predictions_array, chunk_times)
        
        return chunk_predictions_array, chunk_times
        
    except Exception as e:
        print(f"\nError occurred after processing {last_save} samples.")
        print(f"Last successful save was at sample {last_save}")
        print(f"Error: {str(e)}")
        
        # Save whatever we have so far
        if chunk_predictions and chunk_times:
            temp_predictions = np.array(chunk_predictions)
            temp_times = np.array(chunk_times)
            np.save(f'chunk_results/error_chunk{chunk_info["chunk_number"]:03d}_predictions.npy', temp_predictions)
            np.save(f'chunk_results/error_chunk{chunk_info["chunk_number"]:03d}_times.npy', temp_times)
            print(f"Saved partial results up to sample {len(chunk_predictions)}")
        
        raise

# Also modify save_chunk_results to make accuracy optional
def save_chunk_results(chunk_info, predictions, execution_times, accuracy=None):
    """Save results for this chunk with unique filenames"""
    chunk_number = chunk_info['chunk_number']
    chunk_dir = 'chunk_results'
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Save with unique filenames including chunk number
    np.save(f'{chunk_dir}/chunk{chunk_number:03d}_predictions.npy', predictions)
    np.save(f'{chunk_dir}/chunk{chunk_number:03d}_execution_times.npy', execution_times)
    
    # Save metadata
    metadata = {
        'chunk_number': chunk_number,
        'start_idx': chunk_info['start_idx'],
        'end_idx': chunk_info['end_idx'],
        'size': chunk_info['size'],
        'avg_time_per_sample': np.mean(execution_times),
        'total_time': np.sum(execution_times)
    }
    
    with open(f'{chunk_dir}/chunk{chunk_number:03d}_metadata.txt', 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nChunk {chunk_number:03d} Results:")
    print(f"Samples {chunk_info['start_idx']} to {chunk_info['end_idx']} (Size: {chunk_info['size']})")
    print(f"Average time per sample: {metadata['avg_time_per_sample']:.2f} seconds")
    print(f"Total chunk time: {metadata['total_time']:.2f} seconds")

def combine_all_chunks():
    """Combine results from all processed chunks and generate final summary"""
    chunk_dir = 'chunk_results'
    if not os.path.exists(chunk_dir):
        print("No chunk results found!")
        return
    
    all_predictions = []
    all_times = []
    all_accuracies = []
    chunk_metadata = []
    
    # Find all chunk files
    for chunk_num in range(10):  # Assuming 10 chunks (0-9)
        pred_file = f'{chunk_dir}/chunk{chunk_num:03d}_predictions.npy'
        time_file = f'{chunk_dir}/chunk{chunk_num:03d}_execution_times.npy'
        meta_file = f'{chunk_dir}/chunk{chunk_num:03d}_metadata.txt'
        
        if not all(os.path.exists(f) for f in [pred_file, time_file, meta_file]):
            print(f"Warning: Missing files for chunk {chunk_num}")
            continue
            
        # Load data
        predictions = np.load(pred_file)
        times = np.load(time_file)
        
        all_predictions.append(predictions)
        all_times.extend(times)
        
        # Read metadata for accuracy
        with open(meta_file, 'r') as f:
            metadata = {}
            for line in f:
                key, value = line.strip().split(': ')
                metadata[key] = value
            chunk_metadata.append(metadata)
            if 'accuracy' in metadata:
                all_accuracies.append(float(metadata['accuracy']))
    
    # Combine all results
    combined_predictions = np.concatenate(all_predictions)
    
    # Calculate overall statistics
    overall_stats = {
        'total_samples': len(combined_predictions),
        'total_time': sum(all_times),
        'average_time_per_sample': np.mean(all_times),
        'overall_accuracy': np.mean(all_accuracies),
        'processed_chunks': len(all_predictions)
    }
    
    # Save combined results
    np.save(f'{chunk_dir}/combined_predictions.npy', combined_predictions)
    np.save(f'{chunk_dir}/combined_times.npy', np.array(all_times))
    
    # Save overall summary
    with open(f'{chunk_dir}/final_summary.txt', 'w') as f:
        f.write("=== Final Summary ===\n\n")
        f.write(f"Total Samples Processed: {overall_stats['total_samples']}\n")
        f.write(f"Number of Chunks Processed: {overall_stats['processed_chunks']}\n")
        f.write(f"Overall Accuracy: {overall_stats['overall_accuracy']:.4f}\n")
        f.write(f"Total Processing Time: {overall_stats['total_time']:.2f} seconds\n")
        f.write(f"Average Time per Sample: {overall_stats['average_time_per_sample']:.2f} seconds\n\n")
        
        f.write("=== Individual Chunk Results ===\n\n")
        for metadata in chunk_metadata:
            f.write(f"Chunk {metadata['chunk_number']}:\n")
            f.write(f"  Samples: {metadata['start_idx']} to {metadata['end_idx']}\n")
            f.write(f"  Accuracy: {metadata['accuracy']}\n")
            f.write(f"  Avg Time/Sample: {metadata['avg_time_per_sample']} seconds\n\n")
    
    print("\n=== Final Summary ===")
    print(f"Total Samples Processed: {overall_stats['total_samples']}")
    print(f"Number of Chunks Processed: {overall_stats['processed_chunks']}")
    print(f"Overall Accuracy: {overall_stats['overall_accuracy']:.4f}")
    print(f"Total Processing Time: {overall_stats['total_time']:.2f} seconds")
    print(f"Average Time per Sample: {overall_stats['average_time_per_sample']:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Process test set chunks and combine results')
    parser.add_argument('--chunk', type=int, help='Chunk index to process (0-9)')
    parser.add_argument('--combine', action='store_true', help='Combine all processed chunks')
    args = parser.parse_args()
    
    if args.combine:
        print("Combining all processed chunks...")
        combine_all_chunks()
        return
    
    # If no chunk specified, ask for input
    if args.chunk is None:
        while True:
            try:
                chunk_idx = int(input("Enter chunk index to process (0-9): "))
                if 0 <= chunk_idx <= 9:
                    break
                print("Please enter a valid chunk index between 0 and 9")
            except ValueError:
                print("Please enter a valid number")
    else:
        chunk_idx = args.chunk
    
    print(f"Using device: {DEVICE}")
    start_time = time.time()
    
    try:
        # Load all test data
        test_features, test_labels = load_and_preprocess_data(
            'features/Test_Features.npy',
            'features/Test_Targets.npy',
            'features/scaler.joblib'
        )
        
        # Get chunk information
        chunk_info = get_chunk_info(len(test_features), chunk_idx)
        print(f"\nProcessing chunk {chunk_idx} ({chunk_info['start_idx']} to {chunk_info['end_idx']})")
        
        # Load the clients and servers
        clients = []
        servers = []
        print("\nLoading clients and servers...")
        for i in range(NUM_CLASSES):
            client_path = os.path.join(FHE_DIRECTORY, f"classifier_{i}")
            if not os.path.exists(client_path):
                raise FileNotFoundError(f"Directory not found: {client_path}")
            client = FHEModelClient(path_dir=client_path, key_dir=KEY_DIRECTORY)
            server = FHEModelServer(path_dir=client_path)
            clients.append(client)
            servers.append(server)
            print(f"✓ Loaded classifier_{i}")
        
        # Process the specified chunk
        process_chunk(chunk_info, test_features, test_labels, clients, servers)
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# source ~/.virtualenvs/concrete_env/bin/activate
# python3 M/MPerform_inference.py --chunk 11 &> result_M_Perform_inference0.out

# Process individual chunks
# python3 MPerform_inference.py --chunk 0
# python3 MPerform_inference.py --chunk 1
# # ... process all chunks ...

# # Combine all results
# python3 MPerform_inference.py --combine

