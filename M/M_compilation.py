import numpy as np
import torch
import os
import shutil
import time
from tqdm import tqdm
from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
from M_qat_training import EnsembleQuantizedModel, QuantFHEFriendlyBinaryClassifier, NUM_CLASSES
from sklearn.preprocessing import StandardScaler

FHE_DIRECTORY = './fhe_ensemble_client_server_files/'
MODEL_PATH = 'Final_Fhe_friendly_qat_ensemble_model.pth'
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

def compile_binary_classifier(model, input_dim, representative_data):
    print(f"Compiling binary classifier...")
    try:
        # Move model to CPU for compilation
        model = model.cpu()
        fhe_circuit = compile_brevitas_qat_model(
            model,
            representative_data,
            n_bits=6,
            p_error=0.01,
            verbose=True
        )
        return fhe_circuit
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
        train_features = torch.tensor(train_features, dtype=torch.float32, device=DEVICE)
        train_labels = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)
        data_load_time = time.time() - data_load_start
        print(f"Data loading time: {data_load_time:.2f} seconds")
        print(f"Shape of train_features: {train_features.shape}")
        input_dim, num_classes = train_features.shape[1], NUM_CLASSES
        print(f"input_dim: {input_dim}, num_classes: {num_classes}")
        overall_pbar.update(1)

        model_load_start = time.time()
        ensemble_model = EnsembleQuantizedModel(input_dim, num_classes).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        ensemble_model.load_state_dict(state_dict, strict=False)
        ensemble_model.eval()
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {model_load_time:.2f} seconds")
        overall_pbar.update(1)

        compilation_start = time.time()
        fhe_circuits = []
        for i in range(NUM_CLASSES):
            binary_classifier = ensemble_model.models[i]
            # Move data to CPU for compilation
            cpu_features = train_features.cpu().numpy()
            fhe_circuit = compile_binary_classifier(binary_classifier, input_dim, cpu_features)
            fhe_circuits.append(fhe_circuit)
        compilation_time = time.time() - compilation_start
        print(f"Compilation finished in {compilation_time:.2f} seconds")
        overall_pbar.update(1)

        with tqdm(total=100, desc="Saving models") as pbar:
            start_time = time.time()
            for i, fhe_circuit in enumerate(fhe_circuits):
                dev = FHEModelDev(path_dir=os.path.join(FHE_DIRECTORY, f"classifier_{i}"), model=fhe_circuit)
                dev.save()
                pbar.update(100 // NUM_CLASSES)
        save_time = time.time() - start_time
        print(f"FHE models saved to {FHE_DIRECTORY} in {save_time:.2f} seconds")
        overall_pbar.update(1)

        print("Setting up client and generating keys...")
        start_time = time.time()
        client_key_dir = "./keys_client_ensemble"
        os.makedirs(client_key_dir, exist_ok=True)
        clients = []
        for i in range(NUM_CLASSES):
            client = FHEModelClient(path_dir=os.path.join(FHE_DIRECTORY, f"classifier_{i}"), key_dir=client_key_dir)
            client.generate_private_and_evaluation_keys()
            clients.append(client)
        client_setup_time = time.time() - start_time
        print(f"Client setup and key generation completed in {client_setup_time:.2f} seconds")
        overall_pbar.update(1)

        print("Setting up servers...")
        start_time = time.time()
        servers = []
        for i in range(NUM_CLASSES):
            server = FHEModelServer(path_dir=os.path.join(FHE_DIRECTORY, f"classifier_{i}"))
            servers.append(server)
        server_setup_time = time.time() - start_time
        print(f"Server setup completed in {server_setup_time:.2f} seconds")
        overall_pbar.update(1)

        print("Running example prediction...")
        try:
            with tqdm(total=100, desc="Running example prediction") as pbar:
                start_time = time.time()
                X_new = torch.rand(1, input_dim, dtype=torch.float32, device=DEVICE)
                X_new = scaler.transform(X_new.cpu().numpy())
                X_new = torch.tensor(X_new, dtype=torch.float32, device=DEVICE)
                
                results = []
                for i in range(NUM_CLASSES):
                    X_new_cpu = X_new.cpu().numpy()
                    encrypted_data = clients[i].quantize_encrypt_serialize(X_new_cpu)
                    serialized_evaluation_keys = clients[i].get_serialized_evaluation_keys()
                    encrypted_result = servers[i].run(encrypted_data, serialized_evaluation_keys)
                    result = clients[i].deserialize_decrypt_dequantize(encrypted_result)
                    results.append(result)
                    pbar.update(100 // NUM_CLASSES)
                
                final_result = np.argmax(results)
            prediction_time = time.time() - start_time
            print(f"Example prediction completed in {prediction_time:.2f} seconds")
            print("Example prediction:", final_result)
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