import numpy as np
import torch
import os
import shutil
import time
from tqdm import tqdm
from S_qat_training import QuantFHEFriendlyMLPClassifier

FHE_DISABLED_DIRECTORY = './fhe_disabled_files/'
MODEL_PATH = 'Good_Fhe_friendly_qat_mlp_model.pth'

def load_features(file_path):
    return np.load(file_path)

def prepare_model_fhe_disabled(model):
    print("Preparing model with FHE disabled...")
    try:
        with tqdm(total=100, desc="Preparing model (FHE disabled)") as pbar:
            start_time = time.time()
            # No additional preparation needed, as the model is already quantized with Brevitas
            end_time = time.time()
            pbar.update(100)
        preparation_time = end_time - start_time
        print(f"Preparation finished in {preparation_time:.2f} seconds.")
        return model, preparation_time
    except Exception as e:
        print(f"An error occurred during preparation: {str(e)}")
        raise

def main():
    total_start_time = time.time()
    with tqdm(total=5, desc="Overall Progress") as overall_pbar:
        # Clear and recreate directory
        if os.path.exists(FHE_DISABLED_DIRECTORY):
            shutil.rmtree(FHE_DISABLED_DIRECTORY)
        os.makedirs(FHE_DISABLED_DIRECTORY)
        overall_pbar.update(1)

        # Load data and model
        train_features = load_features('features/1TF.npy')
        train_labels = load_features('features/1TT.npy')
        input_dim, num_classes = train_features.shape[1], len(np.unique(train_labels))
        model = QuantFHEFriendlyMLPClassifier(input_dim, num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        overall_pbar.update(1)

        # Prepare model with FHE disabled (no additional preparation needed)
        fhe_disabled_model, _ = prepare_model_fhe_disabled(model)
        overall_pbar.update(1)

        # Save the FHE-disabled model
        torch.save(fhe_disabled_model.state_dict(), os.path.join(FHE_DISABLED_DIRECTORY, 'fhe_disabled_model.pth'))
        overall_pbar.update(1)

        print("Preparation complete for FHE-disabled model.")

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()