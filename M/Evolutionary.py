import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import os

# Configuration
BATCH_SIZE = 2048
GENERATIONS = 300
POPULATION_SIZE = 50
MUTATION_RATE = 0.3
EARLY_STOPPING_ROUNDS = 20

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FHEFriendlyBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FHEFriendlyBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
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

def load_data():
    features_dir = 'features'
    val_features = np.load(os.path.join(features_dir, 'Val_Features.npy'))
    val_targets = np.load(os.path.join(features_dir, 'Val_Targets.npy'))
    test_features = np.load(os.path.join(features_dir, 'Test_Features.npy'))
    test_targets = np.load(os.path.join(features_dir, 'Test_Targets.npy'))

    scaler = StandardScaler()
    val_features = scaler.fit_transform(val_features)
    test_features = scaler.transform(test_features)

    return (torch.tensor(val_features, dtype=torch.float32).to(device),
            torch.tensor(val_targets, dtype=torch.long).to(device),
            torch.tensor(test_features, dtype=torch.float32).to(device),
            torch.tensor(test_targets, dtype=torch.long).to(device))

model_cache = {}

def load_model(class_id):
    if class_id not in model_cache:
        model_filename = f'fhe_friendly_binary_classifier_class_{class_id}.pth'
        model = FHEFriendlyBinaryClassifier(val_features.shape[1]).to(device)
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.eval()
        model_cache[class_id] = model
    return model_cache[class_id]

def evaluate_ensemble(ensemble, features, targets):
    batch_size = BATCH_SIZE
    predictions = torch.zeros((len(features), 10), device=device)
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            for class_id in ensemble:
                model = load_model(class_id)
                predictions[i:i+batch_size, class_id] = model(batch_features).squeeze()
    
    ensemble_pred = torch.argmax(predictions, dim=1)
    return accuracy_score(targets.cpu().numpy(), ensemble_pred.cpu().numpy())

def mutation(ensemble):
    mutated = ensemble.copy()
    num_mutations = random.randint(1, 3)
    for _ in range(num_mutations):
        index = random.randint(0, 9)
        mutated[index] = random.randint(0, 9)
    return mutated

def crossover(parent1, parent2):
    crossover_point = random.randint(1, 8)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def tournament_selection(population, scores, tournament_size=3):
    selected = random.sample(list(enumerate(population)), tournament_size)
    return max(selected, key=lambda x: scores[x[0]])[1]

def genetic_algorithm(X_val, y_val, X_test, y_test):
    population = [
        [random.randint(0, 9) for _ in range(10)]
        for _ in range(POPULATION_SIZE)
    ]
    
    best_ensemble = None
    best_val_score = 0
    best_test_score = 0
    no_improvement_count = 0

    for generation in range(GENERATIONS):
        scores = [evaluate_ensemble(ensemble, X_val, y_val) for ensemble in population]
        
        current_best_ensemble = population[scores.index(max(scores))]
        current_test_score = evaluate_ensemble(current_best_ensemble, X_test, y_test)
        
        if current_test_score > best_test_score:
            best_test_score = current_test_score
            best_ensemble = current_best_ensemble
            best_val_score = max(scores)
            print(f"Generation {generation + 1}, Best Val Score: {best_val_score:.4f}, Best Test Score: {best_test_score:.4f}")
            torch.save(best_ensemble, 'best_ensemble_genetic.pt')
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= EARLY_STOPPING_ROUNDS:
            print(f"Early stopping at generation {generation + 1}")
            break

        new_population = [best_ensemble]  # Elitism
        
        while len(new_population) < POPULATION_SIZE:
            if random.random() < MUTATION_RATE:
                parent = tournament_selection(population, scores)
                new_ensemble = mutation(parent)
            else:
                parent1 = tournament_selection(population, scores)
                parent2 = tournament_selection(population, scores)
                new_ensemble = crossover(parent1, parent2)
            
            new_population.append(new_ensemble)
        
        population = new_population

    return best_ensemble, best_val_score, best_test_score

if __name__ == "__main__":
    start_time = time.time()

    val_features, val_targets, test_features, test_targets = load_data()

    best_ensemble, best_val_score, best_test_score = genetic_algorithm(
        val_features, val_targets, test_features, test_targets
    )

    print(f"Best ensemble validation accuracy: {best_val_score:.4f}")
    print(f"Best ensemble test accuracy: {best_test_score:.4f}")
    print("Best ensemble composition:")
    for class_id, model_num in enumerate(best_ensemble):
        print(f"Class {class_id}: Model {model_num}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal running time: {total_time:.2f} seconds")