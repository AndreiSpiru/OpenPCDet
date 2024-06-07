import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import validation
import torch
import ORA_utils as utils
import os
import psutil
from copy import copy
import logging
from joblib import Parallel, delayed
import pickle

# Unset cuDNN logging environment variables
if 'CUDNN_LOGINFO_DBG' in os.environ:
    del os.environ['CUDNN_LOGINFO_DBG']
if 'CUDNN_LOGDEST_DBG' in os.environ:
    del os.environ['CUDNN_LOGDEST_DBG']


# python3 bayesian_ORA1.py --cfg_file cfgs/kitti_models/pointpillar.yaml --budget 200 --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

logging.basicConfig(filename='bayesian_log1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def black_box_function(indices, data, initial_points, attack_path, args, cfg):
    """
    Black-box function : minimise IoU
    
    Args:
        indices (list or np.ndarray): Indices of selected points in the dataset.
        data (np.ndarray): The subset from which points are selected.
        initial_points (np.ndarray): Initial set of points.
        attack_path (str): Path to save the attack data.
        args: Arguments for the validation function.
        cfg: Configuration for the validation function.
    
    Returns:
        float: The score of the black-box function.
    """
    selected_points = data[indices]
    points = utils.shift_selected_points(initial_points, selected_points, 2)

    attack_path_bin = attack_path[:-3] + "bin"
    print(f"Attack path: {attack_path}")
    try:
        points.astype(np.float32).tofile(attack_path_bin)
    except Exception as e:
        logging.error(f"Failed to write to {attack_path_bin}: {e}")

    scores, _ = validation.detection_iou_custom_dataset(args, cfg, [attack_path])
    print(f"Scores: {scores}")
    return scores[0]

def load_attack_points_from_path(args, cfg):
    """
    Load attack points from specified paths
    
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

def bayesian_optimisation_case(args, cfg, data, initial_points, attack_path):
    """
    Perform Bayesian Optimization
    
    Args:
        args: Arguments containing configurations and budget.
        cfg: Configuration for validation functions.
        data (np.ndarray): The subset from which points are selected.
        initial_points (np.ndarray): Initial set of points.
        attack_path (str): Path to save the attack data.
    
    Returns:
        tuple: The minimum value of the black-box function and the best subset of row indices.
    """
    torch.cuda.init()
    torch.cuda.set_device(0)
    if len(data) < 3:
        all_indices = [index for index in range(len(data))]
        return (black_box_function(all_indices, data, initial_points, attack_path, args, cfg), all_indices)
    budget = min(args.budget, len(data))
    k = budget  # Number of indices to select (assuming budget represents this)

    # Create a named search space
    search_space = [Integer(0, data.shape[0] - 1, name=f'index_{i}') for i in range(k)]

    # Create a closure for the black-box function with additional arguments
    def make_objective(data, initial_points, attack_path, args, cfg):
        def objective(indices):
            indices = list(set(int(i) for i in indices if i < data.shape[0]))
            return black_box_function(indices, data, initial_points, attack_path, args, cfg)
        return objective

    objective_with_args = make_objective(data, initial_points, attack_path, args, cfg)

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
    best_indices = list(set(int(i) for i in result.x if i < data.shape[0]))
    print("Best subset of rows indices:", best_indices)
    print("Minimum value of the black-box function:", result.fun)
    return result.fun, best_indices

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    dataset, initial_points, attack_paths = load_attack_points_from_path(args, cfg)

    # # Perform Bayesian optimization in parallel
    # # 2 parallel procceses is what my machine can handle
    # parallel_results = Parallel(n_jobs=1)(delayed(bayesian_optimisation_case)(args, cfg, data, initial_point, attack_path) 
    #                              for (data, initial_point, attack_path) in zip(dataset, initial_points, attack_paths))
    
    parallel_results = [bayesian_optimisation_case(args, cfg, data, initial_point, attack_path) for (data, initial_point, attack_path) in zip(dataset, initial_points, attack_paths)]
    results, best_indices = zip(*parallel_results)
    best_indices_list = list(best_indices)

    logging.critical(f"Best Indices: {best_indices}")
    logging.critical(f"Final Result: {results}")
    logging.critical(f"Mean: {np.mean(results)}")

    print(results)
    print(np.mean(results))

    np.save("bayesian_VLP_results", results)

    # Save best_indices using pickle
    with open("bayesian_VLP_indices.pkl", "wb") as f:
        pickle.dump(best_indices_list, f)
