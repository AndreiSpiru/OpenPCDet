import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import validation
import torch
import ORA_utils as utils
import os
from copy import copy
import logging
import validation_utils
from sklearn.model_selection import KFold

# Unset cuDNN logging environment variables
if 'CUDNN_LOGINFO_DBG' in os.environ:
    del os.environ['CUDNN_LOGINFO_DBG']
if 'CUDNN_LOGDEST_DBG' in os.environ:
    del os.environ['CUDNN_LOGDEST_DBG'

# Cross-validation IoU BORA

# Command to run the script
# python3 bayesian_ORA1.py --cfg_file cfgs/kitti_models/pointpillar.yaml --budget 200 --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

# Set up logging
logging.basicConfig(filename='bayesian_log1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def black_box_function(indices, datasets, initial_points, attack_path, max_length, args, cfg):
    """
    Black-box function: minimize IoU

    Args:
        indices (list or np.ndarray): Indices of selected points in the dataset.
        datasets (list): List of datasets.
        initial_points (np.ndarray): Initial set of points.
        attack_path (str): Path to save the attack data.
        max_length (int): Maximum length of indices.
        args: Arguments for the validation function.
        cfg: Configuration for the validation function.

    Returns:
        float: The score of the black-box function.
    """
    for idx, (data, initial_points, attack_path) in enumerate(zip(datasets, initial_points, attack_paths)):
        scaled_indices = utils.scale_indices(indices, len(data), max_length)
        points = utils.shift_selected_points(initial_points, data[scaled_indices], 2)

        os.makedirs(os.path.dirname(attack_path), exist_ok=True)
        attack_path_bin = attack_path[:-3] + "bin"
        try:
            points.astype(np.float32).tofile(attack_path_bin)
        except Exception as e:
            print(f"Failed to write to {attack_path_bin}: {e}")

    scores, _ = validation.detection_iou_custom_dataset(args, cfg, attack_paths)
    logging.critical(f"Mean score: {np.mean(scores)}")
    print(f"Mean score: {np.mean(scores)}")
    return np.mean(scores)

def black_box_function_save(individual, datasets, original_points, attack_paths, max_length, args, cfg):
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

    scores, _ = validation.detection_iou_custom_dataset(args, cfg, attack_paths)
    validation_utils.create_or_modify_excel_generic(scores, attack_paths, args.ckpt, type="bayesian", file_path="bayesian_results.xlsx")
    logging.critical(f"Mean score: {np.mean(scores)}")
    print(f"Mean score: {np.mean(scores)}")
    return np.mean(scores)

def load_attack_points_from_path(args, cfg):
    """
    Load attack points from specified paths.

    Args:
        args: Arguments containing data path and other configurations.
        cfg: Configuration for validation functions.

    Returns:
        tuple: A tuple containing datasets, original points, and attack paths.
    """
    datasets = []
    attack_paths = []
    original_points = []
    root_path = args.data_path
    root_attack_path = root_path.replace("0-10", "0-10_bayesian")

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

def bayesian_optimisation_combined(args, cfg, datasets, initial_points, attack_path, max_length):
    """
    Perform Bayesian Optimization.

    Args:
        args: Arguments containing configurations and budget.
        cfg: Configuration for validation functions.
        datasets (list): List of datasets.
        initial_points (list): Initial set of points.
        attack_path (str): Path to save the attack data.
        max_length (int): Maximum length of indices.

    Returns:
        tuple: The minimum value of the black-box function and the best subset of row indices.
    """
    torch.cuda.init()
    torch.cuda.set_device(0)

    budget = min(args.budget, max_length)
    k = budget  # Number of indices to select (assuming budget represents this)

    # Create a named search space
    search_space = [Integer(0, max_length, name=f'index_{i}') for i in range(k)]

    # Create a closure for the black-box function with additional arguments
    def make_objective(datasets, initial_points, attack_path, max_length, args, cfg):
        def objective(indices):
            indices = list(set(int(i) for i in indices if i < max_length))
            return black_box_function(indices, datasets, initial_points, attack_path, max_length, args, cfg)
        return objective

    objective_with_args = make_objective(datasets, initial_points, attack_path, max_length, args, cfg)

    @use_named_args(search_space)
    def wrapped_objective(**kwargs):
        indices = [v for k, v in sorted(kwargs.items())]
        return objective_with_args(indices)

    # Perform Bayesian Optimization
    result = gp_minimize(
        wrapped_objective,      # The objective function
        search_space,           # The search space
        n_calls=100,            # Number of evaluations of `wrapped_objective`
        random_state=42,        # Random state for reproducibility
        acq_func='EI',          # Acquisition function, 'Expected Improvement'
        n_jobs=1                # Parallelize the evaluations
    )

    # Print the best result
    best_indices = list(set(int(i) for i in result.x if i < max_length))
    print("Best subset of rows indices:", best_indices)
    print("Minimum value of the black-box function:", result.fun)
    return result.fun, best_indices

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    dataset, initial_points, attack_paths = load_attack_points_from_path(args, cfg)

    kf = KFold(n_splits=5)
    fold_count = 0
    max_length = max(len(dataset) for dataset in dataset)
    validation_results = []
    for train_index, test_index in kf.split(dataset):
        fold_count += 1
        logging.critical(f"Fold {fold_count} started")

        train_datasets = [dataset[i] for i in train_index]
        test_datasets = [dataset[i] for i in test_index]

        train_points = [initial_points[i] for i in train_index]
        test_points = [initial_points[i] for i in test_index]

        train_paths = [attack_paths[i] for i in train_index]
        test_paths = [attack_paths[i] for i in test_index]

        results, best_indices = bayesian_optimisation_combined(args, cfg, train_datasets, train_points, train_paths, max_length)

        validation_result = black_box_function_save(best_indices, test_datasets, test_points, test_paths, max_length, args, cfg)
        logging.critical(f"Fold done with validation result {validation_result}")
        print(f"Fold done with validation result {validation_result}")
        validation_results.append(validation_result)

    logging.critical(f"Mean: {np.mean(results)}")
    print(results)
    print(np.mean(results))

