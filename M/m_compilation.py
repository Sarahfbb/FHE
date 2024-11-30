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
    print(f"Loading dataset from {features_path} and {labels_path}")
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    return features, labels, scaler

def compile_binary_classifier(model, input_dim, representative_data, class_id):
    print(f"Compiling binary classifier for class {class_id}...")
    try:
        model = model.cpu()
        fhe_circuit = compile_brevitas_qat_model(
            model,
            representative_data,
            n_bits=6,
            p_error=0.01,
            verbose=True
        )
        print(f"Compilation successful for class {class_id}")
        return fhe_circuit
    except Exception as e:
        print(f"An error occurred during compilation for class {class_id}: {str(e)}")
        raise

def main():
    print(f"Using device: {DEVICE}")
    total_start_time = time.time()
    
    try:
        if os.path.exists(FHE_DIRECTORY):
            print(f"Removing existing directory: {FHE_DIRECTORY}")
            shutil.rmtree(FHE_DIRECTORY)
        os.makedirs(FHE_DIRECTORY)
        print(f"Created directory: {FHE_DIRECTORY}")

        data_load_start = time.time()
        train_features, train_labels, scaler = load_dataset('features/Train_Features.npy', 'features/Train_Targets.npy')
        train_features = torch.tensor(train_features, dtype=torch.float32, device=DEVICE)
        train_labels = torch.tensor(train_labels, dtype=torch.long, device=DEVICE)
        data_load_time = time.time() - data_load_start
        print(f"Data loading time: {data_load_time:.2f} seconds")
        print(f"Shape of train_features: {train_features.shape}")
        input_dim, num_classes = train_features.shape[1], NUM_CLASSES
        print(f"input_dim: {input_dim}, num_classes: {num_classes}")

        model_load_start = time.time()
        ensemble_model = EnsembleQuantizedModel(input_dim, num_classes).to(DEVICE)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        ensemble_model.load_state_dict(state_dict, strict=False)
        ensemble_model.eval()
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {model_load_time:.2f} seconds")

        compilation_start = time.time()
        fhe_circuits = []
        for i in range(NUM_CLASSES):
            binary_classifier = ensemble_model.models[i]
            cpu_features = train_features.cpu().numpy()
            fhe_circuit = compile_binary_classifier(binary_classifier, input_dim, cpu_features, i)
            fhe_circuits.append(fhe_circuit)
        compilation_time = time.time() - compilation_start
        print(f"Compilation finished in {compilation_time:.2f} seconds")

        print("Saving models...")
        start_time = time.time()
        for i, fhe_circuit in enumerate(fhe_circuits):
            classifier_dir = os.path.join(FHE_DIRECTORY, f"classifier_{i}")
            os.makedirs(classifier_dir, exist_ok=True)
            dev = FHEModelDev(path_dir=classifier_dir, model=fhe_circuit)
            dev.save()
            print(f"Saved model for classifier_{i} in {classifier_dir}")
        save_time = time.time() - start_time
        print(f"FHE models saved to {FHE_DIRECTORY} in {save_time:.2f} seconds")

        print("Setting up client and generating keys...")
        start_time = time.time()
        client_key_dir = "./keys_client_ensemble"
        os.makedirs(client_key_dir, exist_ok=True)
        clients = []
        for i in range(NUM_CLASSES):
            client_path = os.path.join(FHE_DIRECTORY, f"classifier_{i}")
            client = FHEModelClient(path_dir=client_path, key_dir=client_key_dir)
            client.generate_private_and_evaluation_keys()
            clients.append(client)
            print(f"Generated keys for classifier_{i}")
        client_setup_time = time.time() - start_time
        print(f"Client setup and key generation completed in {client_setup_time:.2f} seconds")

        print("Setting up servers...")
        start_time = time.time()
        servers = []
        for i in range(NUM_CLASSES):
            server_path = os.path.join(FHE_DIRECTORY, f"classifier_{i}")
            server = FHEModelServer(path_dir=server_path)
            servers.append(server)
            print(f"Set up server for classifier_{i}")
        server_setup_time = time.time() - start_time
        print(f"Server setup completed in {server_setup_time:.2f} seconds")

        print("Compilation and setup completed successfully.")

    except Exception as e:
        print(f"An error occurred during the compilation process: {str(e)}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()