import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import KFold
import ORA_utils as utils
import pickle
import validation
import os
import torch
import matplotlib.pyplot as plt
import logging
from copy import copy

# Command to run the script:
# python genetic_ORA_confidence.py --cfg_file cfgs/kitti_models/pointpillar.yaml --budget 200 --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

logging.basicConfig(filename='ga_log_confidence.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_variable_size_datasets(num_datasets, min_examples, max_examples, num_features):
    """
    Generate datasets with a variable number of rows and a fixed number of features.
    Only used for testing purposes
    
    Args:
        num_datasets (int): Number of datasets to create.
        min_examples (int): Minimum number of examples in each dataset.
        max_examples (int): Maximum number of examples in each dataset.
        num_features (int): Number of features in each dataset.
    
    Returns:
        list: List of datasets with variable sizes.
    """
    random_indices = [np.random.rand(random.randint(min_examples, max_examples), num_features) * 100 for _ in range(num_datasets)]
    for arr in random_indices:
        sorted_indices = np.argsort(arr[:, num_features - 1])[::-1]
        arr[:] = arr[sorted_indices]
    return random_indices

def load_attack_points_from_path(root_path, args, cfg):
    """
    Load attack points from the specified path.
    
    Args:
        root_path (str): Path to the root directory containing the datasets.
        args: Command line arguments.
        cfg: Configuration settings.
    
    Returns:
        tuple: Tuple containing datasets, original points, and attack paths.
    """
    datasets = []
    attack_paths = []
    original_points = []
    root_attack_path = root_path.replace("0-10", "0-10_genetic")

    for condition in os.listdir(root_path):
        condition_path = os.path.join(root_path, condition)
        condition_path_attack = os.path.join(root_attack_path, condition)

        if os.path.isdir(condition_path):
            case_args = copy(args)
            case_args.data_path = condition_path
            bboxes, source_file_list = validation.detection_bboxes(case_args, cfg)

            for idx, file_bin in enumerate(source_file_list):
                file_npy = file_bin[:-3] + "npy"
                initial_points = np.load(file_npy)

                bbox = torch.unsqueeze(bboxes[idx], 0)
                bbox = bbox.cpu().numpy()

                points, points_in_bbox, _ = utils.get_point_mask_in_boxes3d(initial_points, bbox)
                non_zero_indices = points_in_bbox.squeeze().nonzero().squeeze()

                if non_zero_indices.numel() == 1:
                    non_zero_indices = torch.tensor([non_zero_indices.item()])

                base_file_npy = os.path.basename(file_npy)
                attack_path = os.path.join(condition_path_attack, base_file_npy)

                original_points.append(points.numpy())
                datasets.append(non_zero_indices.numpy())
                attack_paths.append(attack_path)

    return datasets, original_points, attack_paths

def black_box_loss(selected_data):
    """
    Dummy black-box function. Used for testing
    
    Args:
        selected_data (np.ndarray): Selected data points.
    
    Returns:
        tuple: Sum of the intensities of the selected points.
    """
    return np.sum(selected_data[:, 3]),

def create_unique_individual(max_length, budget):
    """
    Create a unique individual for the genetic algorithm.
    
    Args:
        max_length (int): Maximum length of the individual.
        budget (int): Budget for the attack.
    
    Returns:
        list: Individual for the genetic algorithm.
    """
    budget = min(max_length, budget)
    return creator.Individual(random.sample(range(max_length), budget))

def mutate(individual, max_length, indpb=0.05):
    """
    Enhanced mutation function that avoids repeating values already in the individual.
    
    Args:
        individual (list): Individual to mutate.
        max_length (int): Maximum length of the individual.
        indpb (float): Probability of mutation.
    
    Returns:
        tuple: Mutated individual.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            original_value = individual[i]
            new_value = random.randint(0, max_length - 1)
            attempts = 0
            while new_value in individual:
                new_value = random.randint(0, max_length - 1)
                attempts += 1
                if attempts > 20:
                    break
            if new_value != original_value:
                individual[i] = new_value
    return (individual,)

def crossover(ind1, ind2):
    """
    Perform one-point crossover on two individuals.
    
    Args:
        ind1 (list): First individual.
        ind2 (list): Second individual.
    
    Returns:
        tuple: Two new individuals after crossover.
    """
    child1, child2 = tools.cxOnePoint(ind1, ind2)
    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2

def evaluate(individual, datasets, original_points, attack_paths, max_length, args, cfg):
    """
    Evaluate the fitness of an individual by applying the attack and calculating the detection IOU.
    
    Args:
        individual (list): Individual to evaluate.
        datasets (list): List of datasets.
        original_points (list): List of original points.
        attack_paths (list): List of paths to save attacked files.
        max_length (int): Maximum length of the individual.
        args: Command line arguments.
        cfg: Configuration settings.
    
    Returns:
        tuple: Mean IOU score.
    """
    
    for idx, (data, initial_points, attack_path) in enumerate(zip(datasets, original_points, attack_paths)):
        points = initial_points
        if len(data) != 0:   
            scaled_indices = utils.scale_indices(individual, len(data), max_length)
            points = utils.shift_selected_points(initial_points, data[scaled_indices], 2)

        os.makedirs(os.path.dirname(attack_path), exist_ok=True)
        attack_path_bin = attack_path[:-3] + "bin"
        try:
            points.astype(np.float32).tofile(attack_path_bin)
        except Exception as e:
            print(f"Failed to write to {attack_path_bin}: {e}")

    scores, _ = validation.detection_confidence_custom_dataset(args, cfg, attack_paths)
    logging.critical(f"Mean score: {np.mean(scores)}")
    print(f"Mean score: {np.mean(scores)}")
    return (np.mean(scores),)

def main():
    """
    Main function to run the genetic algorithm for Object Removal Attacks (ORA).
    """
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    args, cfg = validation.parse_config()
    root_path = args.data_path
    datasets, original_points, attack_paths = load_attack_points_from_path(root_path, args, cfg)
    
    max_length = max(dataset.size for dataset in datasets)

    toolbox.register("individual", create_unique_individual, max_length=max_length, budget=args.budget)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, datasets=datasets, original_points=original_points, 
                     attack_paths=attack_paths, max_length=max_length, args=args, cfg=cfg)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate, max_length=max_length, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)

    kf = KFold(n_splits=3)
    results = []
    logging.info("Starting genetic algorithm")
    fold_count = 0

    best_scores = []
    mean_scores = []
    worst_scores = []

    for train_index, test_index in kf.split(datasets):
        logging.critical(f"Fold {fold_count} started")

        train_datasets = [datasets[i] for i in train_index]
        test_datasets = [datasets[i] for i in test_index]

        train_points = [original_points[i] for i in train_index]
        test_points = [original_points[i] for i in test_index]

        train_paths = [attack_paths[i] for i in train_index]
        test_paths = [attack_paths[i] for i in test_index]

        fold_count += 1

        toolbox.register("evaluate", evaluate, datasets=train_datasets, original_points=train_points, 
                     attack_paths=train_paths, max_length=max_length, args=args, cfg=cfg)
        
        for gen in range(50):
            logging.critical(f"Generation {gen} started")
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
            fits = list(map(toolbox.evaluate, offspring))

            fitnesses = []
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
                fitnesses.append(fit)

            best_scores.append(min(fitnesses))
            mean_scores.append(np.mean(fitnesses))
            worst_scores.append(max(fitnesses))

            population = toolbox.select(offspring, len(population))

            logging.critical(f"Generation {gen} completed with best fitness {best_scores[-1]}")
        
        toolbox.register("evaluate", evaluate, datasets=test_datasets, original_points=test_points, 
                     attack_paths=test_paths, max_length=max_length, args=args, cfg=cfg)
        best_ind = tools.selBest(population, 1)[0]
        test_fitness = toolbox.evaluate(best_ind)
        results.append(test_fitness)

    plt.figure(figsize=(10, 5))
    plt.plot(best_scores, label='Best Fitness')
    plt.plot(mean_scores, label='Mean Fitness')
    plt.plot(worst_scores, label='Worst Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.savefig('fitness_over_generations_confidence.png')
    plt.close()

    logging.critical(f"Cross-validation results: {results}")
    logging.critical(f"Best Individual: {best_ind}")
    print("Cross-validation results:", results)
    print("Best Individual:", best_ind)

    with open('best_individual_confidence.pkl', 'wb') as f:
        pickle.dump(best_ind, f)
    logging.info("Saved best individual")

if __name__ == "__main__":
    main()
