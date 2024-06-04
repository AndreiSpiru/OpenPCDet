import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import KFold
import ORA_utils as utils
import validation_utils
import pickle
import validation
import os
import torch
import matplotlib.pyplot as plt
import logging
from pcdet.config import cfg, cfg_from_yaml_file
from copy import copy
import yaml
from easydict import EasyDict

# Unset cuDNN logging environment variables
if 'CUDNN_LOGINFO_DBG' in os.environ:
    del os.environ['CUDNN_LOGINFO_DBG']
if 'CUDNN_LOGDEST_DBG' in os.environ:
    del os.environ['CUDNN_LOGDEST_DBG']

# Command to run the script:
# python3 genetic_ORA_multiple_detectors.py --cfg_file cfgs/kitti_models/pointpillar.yaml --budget 200 --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

logging.basicConfig(filename='ga_log_two_detectors.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

                if non_zero_indices.numel() == 0:
                    non_zero_indices = torch.tensor([])
                
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

# Define elitism function
def elitism(population, offspring, elite_size=5):
    sorted_population = sorted(population, key=lambda ind: ind.fitness.values)
    elite_individuals = sorted_population[:elite_size]
    offspring = sorted(offspring, key=lambda ind: ind.fitness.values)[:len(population) - elite_size]
    offspring.extend(elite_individuals)
    return offspring

def check_overlap_hamming_distance_percentage(arr1, arr2):
    """
    Check if two arrays share more than 75% of their entries.

    Args:
    arr1 (np.array): First array.
    arr2 (np.array): Second array of the same length as arr1.

    Returns:
    bool: True if more than 75% of the entries are the same between the two arrays, False otherwise.
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Both arrays must have the same shape.")

    # Calculate the number of matching entries
    matches = np.sum(arr1 == arr2)

    # Calculate the percentage of matching entries
    percentage = matches / arr1.size

    # Check if the percentage of matching entries is greater than 75%
    return percentage > 0.75

# Define fitness sharing function
def fitness_sharing(individuals):
    for ind in individuals:
        shared_fitness = ind.fitness.values[0]
        count = sum(1 for other in individuals if check_overlap_hamming_distance_percentage(np.array(ind), np.array(other)))
        ind.fitness.values = (shared_fitness * (count**0.01),)
    return individuals

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


def evaluate(individual, datasets, original_points, attack_paths, max_length, first_detector_args, first_detector_cfg, second_detector_args, second_detector_config):
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
        scaled_indices = utils.scale_indices(individual, len(data), max_length)
        points = utils.shift_selected_points(initial_points, data[scaled_indices], 2)

        os.makedirs(os.path.dirname(attack_path), exist_ok=True)
        attack_path_bin = attack_path[:-3] + "bin"
        try:
            points.astype(np.float32).tofile(attack_path_bin)
        except Exception as e:
            print(f"Failed to write to {attack_path_bin}: {e}")

    first_detector_scores, _ = validation.detection_iou_custom_dataset(first_detector_args, first_detector_cfg, attack_paths)


    second_detector_scores, _ = validation.detection_iou_custom_dataset(second_detector_args, second_detector_config, attack_paths)


    scores = first_detector_scores + second_detector_scores
    logging.critical(f"Mean score: {np.mean(scores)}")
    print(f"Mean score: {np.mean(scores)}")
    return (np.mean(scores),)

def evaluate_and_save(individual, datasets, original_points, attack_paths, max_length, first_detector_args, first_detector_cfg, second_detector_args, second_detector_config):
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
        scaled_indices = utils.scale_indices(individual, len(data), max_length)
        points = utils.shift_selected_points(initial_points, data[scaled_indices], 2)

        os.makedirs(os.path.dirname(attack_path), exist_ok=True)
        attack_path_bin = attack_path[:-3] + "bin"
        try:
            points.astype(np.float32).tofile(attack_path_bin)
        except Exception as e:
            print(f"Failed to write to {attack_path_bin}: {e}")


    
    first_detector_scores, _ = validation.detection_iou_custom_dataset(first_detector_args, first_detector_cfg, attack_paths)
    validation_utils.create_or_modify_excel_generic(first_detector_scores, attack_paths, first_detector_scores.ckpt)

    second_detector_scores, _ = validation.detection_iou_custom_dataset(second_detector_args, second_detector_config, attack_paths)
    validation_utils.create_or_modify_excel_generic(second_detector_scores, attack_paths, second_detector_scores.ckpt)

    scores = first_detector_scores + second_detector_scores
    logging.critical(f"Mean score: {np.mean(scores)}")
    print(f"Mean score: {np.mean(scores)}")
    return (np.mean(scores),)
   

def inject_random_individuals(population, toolbox, n=5):
    for _ in range(n):
        new_individual = toolbox.individual()
        population.append(new_individual)
        
def main():
    """
    Main function to run the genetic algorithm for Object Removal Attacks (ORA).
    """
    population_size = 50    

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    second_detector_config = copy(cfg)

    first_detector_args, first_detector_cfg = validation.parse_config()
    root_path = first_detector_args.data_path
    datasets, original_points, attack_paths = load_attack_points_from_path(root_path, first_detector_args, first_detector_cfg)

    second_detector_args = copy(first_detector_args)
    second_detector_args.cfg_file = "cfgs/kitti_models/second.yaml"
    second_detector_args.ckpt = "second_7862.pth"
   
    cfg_from_yaml_file(second_detector_args.cfg_file, second_detector_config)
    
    
    max_length = max(len(dataset) for dataset in datasets)

    toolbox.register("individual", create_unique_individual, max_length=max_length, budget=first_detector_args.budget)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, datasets=datasets, original_points=original_points, 
                     attack_paths=attack_paths, max_length=max_length, first_detector_args=first_detector_args, first_detector_cfg=first_detector_cfg, second_detector_args=second_detector_args,
                     second_detector_config=second_detector_config)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate, max_length=max_length)
    toolbox.register("select", tools.selTournament, tournsize=3)


    population = toolbox.population(n=population_size)

    # Evaluate all individuals in the initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    kf = KFold(n_splits=5)
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
                     attack_paths=train_paths, max_length=max_length, first_detector_args=first_detector_args, first_detector_cfg=first_detector_cfg, second_detector_args=second_detector_args,
                     second_detector_config=second_detector_config)
        
        for gen in range(30):
            logging.critical(f"Generation {gen} started")
            # Selection
            parents = toolbox.select(population, len(population))

            offspring = algorithms.varOr(parents, toolbox, lambda_=len(population), cxpb=0.5, mutpb=0.2)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            offspring = fitness_sharing(offspring)
            
            population = elitism(population, offspring, elite_size=5)  # Apply elitism here

            best_scores.append(min(ind.fitness.values[0] for ind in population))
            mean_scores.append(np.mean([ind.fitness.values[0] for ind in population]))
            worst_scores.append(max(ind.fitness.values[0] for ind in population))

            logging.critical(f"Generation {gen} completed with best fitness {best_scores[-1]}")
        
        toolbox.register("evaluate", evaluate_and_save, datasets=test_datasets, original_points=test_points, 
                     attack_paths=test_paths, max_length=max_length, first_detector_args=first_detector_args, first_detector_cfg=first_detector_cfg, second_detector_args=second_detector_args,
                     second_detector_config=second_detector_config)
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
    plt.savefig('fitness_over_generations.png')
    plt.close()

    logging.critical(f"Cross-validation results: {results}")
    logging.critical(f"Best Individual: {best_ind}")
    print("Cross-validation results:", results)
    print("Best Individual:", best_ind)

    with open('best_individual.pkl', 'wb') as f:
        pickle.dump(best_ind, f)
    logging.info("Saved best individual")

if __name__ == "__main__":
    main()
