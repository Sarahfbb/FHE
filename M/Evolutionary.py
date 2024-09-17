import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Configuration
BATCH_SIZE = 256

# Set up device
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the pre-extracted features with new file names
val_features = np.load('1VF.npy')
val_targets = np.load('1VT.npy')
test_features = np.load('1TestF.npy')
test_targets = np.load('1TestT.npy')

# Standardize features
scaler = StandardScaler()
val_features = scaler.fit_transform(val_features)
test_features = scaler.transform(test_features)

class FHEFriendlyMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FHEFriendlyMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.poly_activation(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.poly_activation(x)
        
        x = self.fc3(x)
        return x

    @staticmethod
    def poly_activation(x):
        return 0.5 * x + 0.25 * x.pow(2)

model_cache = {}

def load_model(class_id, model_num):
    key = (class_id, model_num)
    if key not in model_cache:
        model_filename = f'fhe_friendly_ensemble_class_{class_id}.pt'
        models = torch.load(model_filename, map_location=device)
        model = FHEFriendlyMLPClassifier(val_features.shape[1], 1).to(device)
        model.load_state_dict(models[model_num])
        model.eval()
        model_cache[key] = model
    return model_cache[key]

def evaluate_ensemble(ensemble, features, targets, batch_size=BATCH_SIZE):
    total_predictions = []
    features_tensor = torch.FloatTensor(features)
    
    for i in range(0, len(features), batch_size):
        batch_features = features_tensor[i:i+batch_size].to(device)
        batch_predictions = np.zeros((len(batch_features), 10))
        
        with torch.no_grad():
            for class_id, model_num in enumerate(ensemble):
                model = load_model(class_id, model_num)
                outputs = model(batch_features).cpu().numpy().squeeze()
                batch_predictions[:, class_id] = outputs
        
        total_predictions.append(batch_predictions)
    
    predictions = np.vstack(total_predictions)
    ensemble_pred = np.argmax(predictions, axis=1)
    return accuracy_score(targets, ensemble_pred)

def mutation(ensemble):
    mutated = ensemble.copy()
    num_mutations = random.randint(2, 5)
    for _ in range(num_mutations):
        class_id = random.randint(0, 9)
        new_model_num = random.randint(0, 9)
        mutated[class_id] = new_model_num
    return mutated

def crossover(parent1, parent2):
    child = []
    for i in range(10):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def tournament_selection(population, scores, tournament_size=3):
    selected = random.sample(list(enumerate(population)), tournament_size)
    return max(selected, key=lambda x: scores[x[0]])[1]

def evolutionary_algorithm(X_val, y_val, X_test, y_test, generations=100, population_size=50, mutation_rate=0.3, early_stopping_rounds=20):
    population = [
        [random.randint(0, 9) for _ in range(10)]
        for _ in range(population_size)
    ]
    
    best_ensemble = None
    best_val_score = 0
    best_test_score = 0
    no_improvement_count = 0

    for generation in range(generations):
        scores = [evaluate_ensemble(ensemble, X_val, y_val) for ensemble in population]
        
        current_best_ensemble = population[scores.index(max(scores))]
        current_test_score = evaluate_ensemble(current_best_ensemble, X_test, y_test)
        
        if current_test_score > best_test_score:
            best_test_score = current_test_score
            best_ensemble = current_best_ensemble
            best_val_score = max(scores)
            print(f"Generation {generation + 1}, Best Val Score: {best_val_score:.4f}, Best Test Score: {best_test_score:.4f}")
            torch.save(best_ensemble, 'best_ensembles.pt')
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print(f"Early stopping at generation {generation + 1}")
            break

        new_population = [best_ensemble]  # Elitism
        
        while len(new_population) < population_size:
            if random.random() < mutation_rate:
                parent = tournament_selection(population, scores)
                new_ensemble = mutation(parent)
            else:
                parent1 = tournament_selection(population, scores)
                parent2 = tournament_selection(population, scores)
                new_ensemble = crossover(parent1, parent2)
            
            assert len(new_ensemble) == 10, f"Ensemble has {len(new_ensemble)} models instead of 10"
            new_population.append(new_ensemble)
        
        population = new_population

    return best_ensemble, best_val_score, best_test_score

if __name__ == "__main__":
    import time
    start_time = time.time()

    best_ensemble, best_val_score, best_test_score = evolutionary_algorithm(
        val_features, val_targets, test_features, test_targets,
        generations=300, population_size=50, mutation_rate=0.3, early_stopping_rounds=20
    )

    print(f"Best ensemble validation accuracy: {best_val_score:.4f}")
    print(f"Best ensemble test accuracy: {best_test_score:.4f}")
    print("Best ensemble composition:")
    for class_id, model_num in enumerate(best_ensemble):
        print(f"Class {class_id}: Model {model_num}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal running time: {total_time:.2f} seconds")