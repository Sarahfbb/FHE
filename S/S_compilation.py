import numpy
import torch
import os
import shutil
import time
from tqdm import tqdm
from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
from S_qat_training import QuantFHEFriendlyMLPClassifier

FHE_DIRECTORY = './fhe_client_server_files/'
MODEL_PATH = 'Good_Fhe_friendly_qat_mlp_model.pth'

def load_features(file_path):
    return numpy.load(file_path)

def compile_model(model, input_dim):
    print(f"Type of model: {type(model)}")
    print(f"Type of input_dim: {type(input_dim)}")
    dummy_input = numpy.random.randn(1, input_dim).astype(numpy.float32)
    print(f"Type of dummy_input: {type(dummy_input)}")
    print(f"Shape of dummy_input: {dummy_input.shape}")
    print("Starting compilation...")
    try:
        with tqdm(total=100, desc="Compiling model") as pbar:
            start_time = time.time()
            fhe_circuit = compile_brevitas_qat_model(
                model,
                dummy_input,
                n_bits=6,
                p_error=0.01, #add this
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
    total_start_time = time.time()
    with tqdm(total=7, desc="Overall Progress") as overall_pbar:
        # If the directory already exists, remove it
        if os.path.exists(FHE_DIRECTORY):
            shutil.rmtree(FHE_DIRECTORY)
        os.makedirs(FHE_DIRECTORY)
        overall_pbar.update(1)

        data_load_start = time.time()
        train_features = load_features('features/1TF.npy')
        train_labels = load_features('features/1TT.npy')
        data_load_time = time.time() - data_load_start
        print(f"Data loading time: {data_load_time:.2f} seconds")
        print(f"Type of train_features: {type(train_features)}")
        print(f"Shape of train_features: {train_features.shape}")
        input_dim, num_classes = train_features.shape[1], len(numpy.unique(train_labels))
        print(f"input_dim: {input_dim}, num_classes: {num_classes}")
        overall_pbar.update(1)

        model_load_start = time.time()
        model = QuantFHEFriendlyMLPClassifier(input_dim, num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {model_load_time:.2f} seconds")
        print("Quantized model loaded successfully.")
        overall_pbar.update(1)

        fhe_circuit, compilation_time = compile_model(model, input_dim)
        overall_pbar.update(1)

        # Save the compiled model
        with tqdm(total=100, desc="Saving model") as pbar:
            start_time = time.time()
            dev = FHEModelDev(path_dir=FHE_DIRECTORY, model=fhe_circuit)
            dev.save()
            end_time = time.time()
            pbar.update(100)
        save_time = end_time - start_time
        print(f"FHE model saved to {FHE_DIRECTORY} in {save_time:.2f} seconds")
        overall_pbar.update(1)


        # Setup the client and generate keys
        print("Setting up client and generating keys...")
        start_time = time.time()
        client_key_dir = "./keys_client"
        if not os.path.exists(client_key_dir):
            os.makedirs(client_key_dir)
        client = FHEModelClient(path_dir=FHE_DIRECTORY, key_dir=client_key_dir)
        client.generate_private_and_evaluation_keys()
        serialized_evaluation_keys = client.get_serialized_evaluation_keys()
        end_time = time.time()
        client_setup_time = end_time - start_time
        print(f"Client setup and key generation completed in {client_setup_time:.2f} seconds")
        overall_pbar.update(1)

        # Modify the server setup section:

        # Setup the server
        print("Setting up server...")
        start_time = time.time()
        server = FHEModelServer(path_dir=FHE_DIRECTORY)
        end_time = time.time()
        server_setup_time = end_time - start_time
        print(f"Server setup completed in {server_setup_time:.2f} seconds")
        overall_pbar.update(1)

        # Wrap the example prediction in a try-except block:

        print("Running example prediction...")
        try:
            with tqdm(total=100, desc="Running example prediction") as pbar:
                start_time = time.time()
                X_new = numpy.random.rand(1, input_dim).astype(numpy.float32)
                encrypted_data = client.quantize_encrypt_serialize(X_new)
                pbar.update(33)
                encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
                pbar.update(33)
                result = client.deserialize_decrypt_dequantize(encrypted_result)
                end_time = time.time()
                pbar.update(34)
            prediction_time = end_time - start_time
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
