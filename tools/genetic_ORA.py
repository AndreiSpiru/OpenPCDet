import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import KFold
import pickle

# Function to generate datasets with variable number of rows
def create_variable_size_datasets(num_datasets, min_examples, max_examples, num_features):
    random_indices = [np.random.rand(random.randint(min_examples, max_examples), num_features) * 100 for _ in range(num_datasets)]
    # # Sort each array in the list based on the specified column
    # for arr in random_indices:
    #     # Get the indices that would sort each row based on the specified column
    #     sorted_indices = np.argsort(arr[:, num_features - 1])
    #     # Apply these indices to reorder the rows within the array
    #     arr[:] = arr[sorted_indices]
    return random_indices

def black_box_loss(selected_data):
    return np.sum(selected_data[:, 3]),

def create_unique_individual(max_length):
    return creator.Individual(random.sample(range(max_length), 10))

def mutate(individual, max_length, indpb=0.05):
    for i in range(len(individual)):
        if random.random() < indpb:
            new_index = random.randint(0, max_length - 1)
            while new_index in individual:
                new_index = random.randint(0, max_length - 1)
            individual[i] = new_index
    return individual,

def crossover(ind1, ind2):
    child1, child2 = tools.cxOnePoint(ind1, ind2)
    return ind1, ind2

def evaluate(individual, datasets, max_length):
    scores = []
    count_valid = 0
    for data in datasets:
        scaled_indices = [int(idx * len(data) / max_length) for idx in individual if idx * len(data) / max_length < len(data)]
        if scaled_indices:
            selected_data = data[scaled_indices]
            score = black_box_loss(selected_data)[0]
            count_valid += len(scaled_indices)
            scores.append(score)
    return (np.sum(scores) / count_valid,) if count_valid > 0 else (float('inf'),)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
datasets = create_variable_size_datasets(200, 50, 150, 4)
max_length = max(len(dataset) for dataset in datasets)

toolbox.register("individual", create_unique_individual, max_length=max_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate, datasets=datasets, max_length=max_length)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate, max_length=max_length, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize the population once
population = toolbox.population(n=50)

# Running Genetic Algorithm with Cross-validation
kf = KFold(n_splits=3)
results = []

for train_index, test_index in kf.split(datasets):
    train_sets = [datasets[i] for i in train_index]
    test_sets = [datasets[i] for i in test_index]
    
    # Re-register the 'evaluate' function with current training datasets
    toolbox.register("evaluate", evaluate, datasets=train_sets, max_length=max_length)
    
    for gen in range(40):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        # Correctly apply the evaluate function over the offspring
        fits = list(map(toolbox.evaluate, offspring))  # Use list to consume the map object if needed
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, len(population))
    
    # Re-register for evaluation on test sets
    toolbox.register("evaluate", evaluate, datasets=test_sets, max_length=max_length)
    best_ind = tools.selBest(population, 1)[0]
    test_fitness = toolbox.evaluate(best_ind)
    results.append(test_fitness)

print("Cross-validation results:", results)
print("Best Indiovidual", best_ind)
# Save the best individual from the last fold
with open('best_individual.pkl', 'wb') as f:
    pickle.dump(best_ind, f)
