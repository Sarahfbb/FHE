import numpy as np
import torch
import os
import shutil
import time
from tqdm import tqdm
from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
from S_qat_training import QuantFHEFriendlyMLPClassifier
from sklearn.preprocessing import StandardScaler

FHE_DIRECTORY = './fhe_client_server_files/'
MODEL_PATH = 'Final_Fhe_friendly_qat_mlp_model.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(features_path, labels_path, scaler=None):
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    return features, labels, scaler

def compile_model(model, input_dim, features_path, labels_path):
    print(f"Type of model: {type(model)}")
    print(f"Type of input_dim: {type(input_dim)}")
    
    representative_data, _, _ = load_dataset(features_path, labels_path)
    representative_data = representative_data.astype(np.float32)
    print(f"Shape of representative data: {representative_data.shape}")
    
    print("Starting compilation...")
    try:
        with tqdm(total=100, desc="Compiling model") as pbar:
            start_time = time.time()
            # Ensure model is on CPU for compilation
            model = model.cpu()
            fhe_circuit = compile_brevitas_qat_model(
                model,
                representative_data,
                n_bits=6,
                p_error=0.01,
                verbose=True
            )
            end_time = time.time()
            pbar.update(100)
        compilation_time = end_time - start_time
        print(f"Compilation finished in {compilation_time:.2f} seconds. Type of fhe_circuit: {type(fhe_circuit)}")
        return fhe_circuit, compilation_time
    except Exception as e:
        print(f"An error occurred during compilation: {str(e)}")
        raise

def main():
    print(f"Using device: {DEVICE}")
    total_start_time = time.time()
    with tqdm(total=7, desc="Overall Progress") as overall_pbar:
        if os.path.exists(FHE_DIRECTORY):
            shutil.rmtree(FHE_DIRECTORY)
        os.makedirs(FHE_DIRECTORY)
        overall_pbar.update(1)

        data_load_start = time.time()
        train_features, train_labels, scaler = load_dataset('features/Train_Features.npy', 'features/Train_Targets.npy')
        data_load_time = time.time() - data_load_start
        print(f"Data loading time: {data_load_time:.2f} seconds")
        print(f"Shape of train_features: {train_features.shape}")
        input_dim, num_classes = train_features.shape[1], len(np.unique(train_labels))
        print(f"input_dim: {input_dim}, num_classes: {num_classes}")
        overall_pbar.update(1)

        model_load_start = time.time()
        model = QuantFHEFriendlyMLPClassifier(input_dim, num_classes).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {model_load_time:.2f} seconds")
        overall_pbar.update(1)

        fhe_circuit, compilation_time = compile_model(
            model, 
            input_dim, 
            'features/Train_Features.npy', 
            'features/Train_Targets.npy'
        )
        overall_pbar.update(1)

        with tqdm(total=100, desc="Saving model") as pbar:
            start_time = time.time()
            dev = FHEModelDev(path_dir=FHE_DIRECTORY, model=fhe_circuit)
            dev.save()
            end_time = time.time()
            pbar.update(100)
        save_time = end_time - start_time
        print(f"FHE model saved to {FHE_DIRECTORY} in {save_time:.2f} seconds")
        overall_pbar.update(1)

        print("Setting up client and generating keys...")
        start_time = time.time()
        client_key_dir = "./keys_client"
        os.makedirs(client_key_dir, exist_ok=True)
        client = FHEModelClient(path_dir=FHE_DIRECTORY, key_dir=client_key_dir)
        client.generate_private_and_evaluation_keys()
        serialized_evaluation_keys = client.get_serialized_evaluation_keys()
        client_setup_time = time.time() - start_time
        print(f"Client setup and key generation completed in {client_setup_time:.2f} seconds")
        overall_pbar.update(1)

        print("Setting up server...")
        start_time = time.time()
        server = FHEModelServer(path_dir=FHE_DIRECTORY)
        server_setup_time = time.time() - start_time
        print(f"Server setup completed in {server_setup_time:.2f} seconds")
        overall_pbar.update(1)

        print("Running example prediction...")
        try:
            with tqdm(total=100, desc="Running example prediction") as pbar:
                start_time = time.time()
                X_new = np.random.rand(1, input_dim).astype(np.float32)
                X_new = scaler.transform(X_new)
                X_new_tensor = torch.tensor(X_new, dtype=torch.float32, device=DEVICE)
                # Move to CPU for FHE operations
                X_new_cpu = X_new_tensor.cpu().numpy()
                encrypted_data = client.quantize_encrypt_serialize(X_new_cpu)
                pbar.update(33)
                encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
                pbar.update(33)
                result = client.deserialize_decrypt_dequantize(encrypted_result)
                pbar.update(34)
            prediction_time = time.time() - start_time
            print(f"Example prediction completed in {prediction_time:.2f} seconds")
            print("Example prediction:", result)
        except Exception as e:
            print(f"An error occurred during the example prediction: {str(e)}")

    total_time = time.time() - total_start_time
    print("\nTotal running times:")
    print(f"Data loading: {data_load_time:.2f} seconds")
    print(f"Model loading: {model_load_time:.2f} seconds")
    print(f"Model compilation: {compilation_time:.2f} seconds")
    print(f"Model saving: {save_time:.2f} seconds")
    print(f"Client setup: {client_setup_time:.2f} seconds")
    print(f"Server setup: {server_setup_time:.2f} seconds")
    print(f"Example prediction: {prediction_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()