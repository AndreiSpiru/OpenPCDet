import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import KFold
import ORA_utils as utils
import pickle
import validation
import os
import torch
from copy import copy

#python genetic_ORA.py --cfg_file cfgs/kitti_models/pointpillar.yaml    --budget 200 --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

# Function to generate datasets with variable number of rows
def create_variable_size_datasets(num_datasets, min_examples, max_examples, num_features):
    random_indices = [np.random.rand(random.randint(min_examples, max_examples), num_features) * 100 for _ in range(num_datasets)]
    # Sort each array in the list based on the specified column
    for arr in random_indices:
        # Get the indices that would sort each row based on the specified column
        sorted_indices = np.argsort(arr[:, num_features - 1])[:: -1]
        # Apply these indices to reorder the rows within the array
        arr[:] = arr[sorted_indices]
    return random_indices

def load_attack_points_from_path(root_path, args, cfg):
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
            #print(bboxes)
            for idx, file_bin in enumerate(source_file_list):
                #if TMP_ok == False:
                    file_npy = file_bin[: -3] + "npy"
                    initial_points = np.load(file_npy)
    
                    bbox = torch.unsqueeze(bboxes[idx], 0)
                    bbox = bbox.cpu().numpy()  

                    points, points_in_bbox, _ = utils.get_point_mask_in_boxes3d(initial_points, bbox)
                    non_zero_indices = points_in_bbox.squeeze().nonzero().squeeze()
                    points_in_bbox = points[non_zero_indices].numpy()
                    
                    sorted_indices = np.argsort(points_in_bbox[:, 3])

                    # print(f"inainte {non_zero_indices}")
                    # print(f"nou {non_zero_indices[sorted_indices]}")
                    # points_in_bbox = points_in_bbox[sorted_indices]
                    #print(points_in_bbox[sorted_indices])

                    base_file_npy = os.path.basename(file_npy)
                    attack_path = os.path.join(condition_path_attack, base_file_npy)
                    

                    original_points.append(points.numpy())
                    datasets.append(non_zero_indices[sorted_indices].numpy())
                    attack_paths.append(attack_path)
    
    #print(datasets)
    return datasets, original_points, attack_paths

def black_box_loss(selected_data):
    return np.sum(selected_data[:, 3]),

def create_unique_individual(max_length, budget):
    budget = min(max_length, budget)
    return creator.Individual(random.sample(range(max_length), budget))

def mutate(individual, max_length, indpb=0.05):
    for i in range(len(individual)):
        if random.random() < indpb:
            original_value = individual[i]
            while True:
                new_index = random.randint(0, max_length - 1)
                if new_index != original_value:
                    individual[i] = new_index
                    break
    return individual,


def crossover(ind1, ind2):
    # This function performs a one-point crossover at a random position along the list of attributes.
    # The child1 and child2 created from this operation should be the ones returned.
    child1, child2 = tools.cxOnePoint(ind1, ind2)
    # You need to update ind1 and ind2 to be the children, not the original parents.
    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2


def evaluate(individual, datasets, original_points, attack_paths, max_length, args, cfg):
    # Save attacked files in their respective directories
    for idx, (data, initial_points, attack_path) in enumerate(zip(datasets, original_points, attack_paths)):
        scaled_indices = [int(idx * len(data) / max_length) for idx in individual if idx * len(data) / max_length < len(data)]  
        points = utils.shift_selected_points(initial_points, data[scaled_indices], 2)
        # differing_rows = 0
        # for row1, row2 in zip(points, initial_points):
        #     if not np.array_equal(row1, row2):
        #         differing_rows += 1

        # print(differing_rows)
        os.makedirs(os.path.dirname(attack_path), exist_ok=True)
        attack_path_bin = attack_path[:-3] + "bin"
        try:
            points.astype(np.float32).tofile(attack_path_bin)
        except Exception as e:
            print(f"Failed to write to {attack_path_bin}: {e}")
    
    
    scores, _ = validation.detection_iou_custom_dataset(args, cfg, attack_paths)
    #print(f"All scores {scores}")
    print(f"Mean score:{np.mean(scores)}")
    return (np.mean(scores),)

    
def main():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    args, cfg = validation.parse_config()
    root_path = args.data_path
    print(root_path)
    datasets, original_points, attack_paths = load_attack_points_from_path(root_path, args, cfg)
    #datasets = create_variable_size_datasets(200, 50, 150, 4)
    
    max_length = max(len(dataset) for dataset in datasets)

    toolbox.register("individual", create_unique_individual, max_length=max_length, budget = args.budget)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, datasets=datasets, original_points=original_points, 
                     attack_paths=attack_paths, max_length=max_length, args=args, cfg=cfg)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate, max_length=max_length, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize the population once
    population = toolbox.population(n=30)

    # Running Genetic Algorithm with Cross-validation
    kf = KFold(n_splits=3)
    results = []

    fold_count = 0
    for train_index, test_index in kf.split(datasets):
        print(f"Fold {fold_count} started")
        train_datasets = [datasets[i] for i in train_index]
        test_datasets = [datasets[i] for i in test_index]

        train_points = [original_points[i] for i in train_index]
        test_points = [original_points[i] for i in test_index]

        train_paths = [attack_paths[i] for i in train_index]
        test_paths = [attack_paths[i] for i in test_index]

        fold_count += 1
        print(train_index)
        print(test_index)
        # Re-register the 'evaluate' function with current training datasets
        toolbox.register("evaluate", evaluate, datasets=train_datasets, original_points=train_points, 
                     attack_paths=train_paths, max_length=max_length, args=args, cfg=cfg)
        
        
        for gen in range(30):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
            # Correctly apply the evaluate function over the offspring
            fits = list(map(toolbox.evaluate, offspring))  # Use list to consume the map object if needed
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, len(population))
            print(f"Gen {gen} done")
        
        # Re-register for evaluation on test sets
        toolbox.register("evaluate", evaluate, datasets=test_datasets, original_points=test_points, 
                     attack_paths=test_paths, max_length=max_length, args=args, cfg=cfg)
        best_ind = tools.selBest(population, 1)[0]
        test_fitness = toolbox.evaluate(best_ind)
        results.append(test_fitness)
        print(f"Fold {fold_count} done")
        print("------------------------")

    print("Cross-validation results:", results)
    print("Best Individual", best_ind)
    # Save the best individual from the last fold
    with open('best_individual.pkl', 'wb') as f:
        pickle.dump(best_ind, f)

if __name__ == "__main__":
    main()
