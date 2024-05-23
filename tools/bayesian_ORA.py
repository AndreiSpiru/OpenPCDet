import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import validation
import torch
import ORA_utils as utils
import os
from copy import copy
import logging

# can be removed
logging.basicConfig(filename='bayesian_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulated black-box function: Sum of selected points' intensities
def black_box_function(indices, data, initial_points, attack_path, args, cfg):
    indices = np.clip(np.round(indices).astype(int), 0, len(data) - 1)
    indices = np.unique(indices)  # Ensure indices are unique
    selected_points = data[indices]

    points = utils.shift_selected_points(initial_points, selected_points, 2)

    attack_path_bin = attack_path[:-3] + "bin"
    logging.critical(f"Attack path: {attack_path}")
    try:
        points.astype(np.float32).tofile(attack_path_bin)
    except Exception as e:
        logging.error(f"Failed to write to {attack_path_bin}: {e}")

    scores, _ = validation.detection_iou_custom_dataset(args, cfg, [attack_path])
    logging.critical(f"Scores: {scores}")
    return scores[0]

# Acquisition function to guide the Bayesian Optimization
def acquisition_function(x, model, data, n_select):
    x = np.clip(np.round(x).astype(int), 0, len(data) - 1)
    x = np.unique(x)  # Ensure indices are unique
    if len(x) < n_select:
        return 1e6  # Penalize for insufficient indices
    x = x[:n_select]  # Ensure we use only `n_select` indices
    x = x.reshape(-1, 1)
    mu, sigma = model.predict(x, return_std=True)
    logging.debug(f"Acquisition function evaluated at {x}: mu={mu}, sigma={sigma}")
    return -(mu - 1.96 * sigma).sum()  # Minimize the negative sum of LCB

def load_attack_points_from_path(args, cfg):
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
                points_in_bbox = points[non_zero_indices].numpy()

                sorted_indices = np.argsort(points_in_bbox[:, 3])

                base_file_npy = os.path.basename(file_npy)
                attack_path = os.path.join(condition_path_attack, base_file_npy)

                original_points.append(points.numpy())
                datasets.append(non_zero_indices[sorted_indices].numpy())
                attack_paths.append(attack_path)

    return datasets, original_points, attack_paths

def delete_file(file_path):
    try:
        os.remove(file_path)
        logging.critical(f"File '{file_path}' deleted successfully.")
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found.")
    except PermissionError:
        logging.error(f"Permission denied to delete file '{file_path}'.")

def bayesian_optimisation_case(args, cfg, data, initial_points, attack_path):
    length_scale_bounds = (1e-2, 1e7)  # Further increase the upper bound

    kernel = Matern(length_scale=1.0, length_scale_bounds=length_scale_bounds)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=10)

    budget = min(args.budget, len(data))
    k = budget  # Number of indices to select (assuming budget represents this)
    initial_indices = np.random.choice(len(data), k, replace=False).reshape(-1, 1)
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

logging.basicConfig(filename='bayesian_log1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulated black-box function: Sum of selected points' intensities
def black_box_function(indices, data, initial_points, attack_path, args, cfg):
    selected_points = data[indices]

    points = utils.shift_selected_points(initial_points, selected_points, 2)

    # process = psutil.Process(os.getpid())
    # attack_path = attack_path[:-4] + str(process.cpu_num()) + ".npy"
    attack_path_bin = attack_path[:-3] + "bin"
    logging.critical(f"Attack path: {attack_path}")
    try:
        points.astype(np.float32).tofile(attack_path_bin)
    except Exception as e:
        logging.error(f"Failed to write to {attack_path_bin}: {e}")

    scores, _ = validation.detection_iou_custom_dataset(args, cfg, [attack_path])
    logging.critical(f"Scores: {scores}")
    return scores[0]

def load_attack_points_from_path(args, cfg):
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
                points_in_bbox = points[non_zero_indices].numpy()

                sorted_indices = np.argsort(points_in_bbox[:, 3])

                base_file_npy = os.path.basename(file_npy)
                attack_path = os.path.join(condition_path_attack, base_file_npy)

                original_points.append(points.numpy())
                datasets.append(non_zero_indices[sorted_indices].numpy())
                attack_paths.append(attack_path)

    return datasets, original_points, attack_paths

def bayesian_optimisation_case(args, cfg, data, initial_points, attack_path):
    
    budget = min(args.budget, len(data))
    k = budget  # Number of indices to select (assuming budget represents this)
    # Create a named search space
    search_space = [Integer(0, data.shape[0] - 1, name=f'index_{i}') for i in range(k)]
    
    # Create a closure for the black-box function with additional arguments
    def make_objective(data, initial_points, attack_path, args, cfg):
        def objective(indices):
            # Convert indices to integers and remove duplicates
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
        acq_func='EI',           # Acquisition function, 'Expected Improvement'
        n_jobs=1               # Parallelize the evaluations
    )

    # Print the best result
    best_indices = list(set(int(i) for i in result.x if i < data.shape[0]))
    print("Best subset of rows indices:", best_indices)
    print("Minimum value of the black-box function:", result.fun)
    return result.fun, best_indices

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    dataset, initial_points, attack_paths = load_attack_points_from_path(args, cfg)
    #results = bayesian_optimisation_case(args, cfg, dataset[1], initial_points[1], attack_paths[1])
    parallel_results = Parallel(n_jobs=2)(delayed(bayesian_optimisation_case)(args, cfg, data, initial_point, attack_path) 
                                 for (data, initial_point, attack_path) in zip(dataset, initial_points, attack_paths))
    results, best_indices = zip(*parallel_results)
    best_indices_list = list(best_indices)
    logging.critical(f"Best Indices: {best_indices}")
    logging.critical(f"Final Result: {results}")
    logging.critical(f"Mean: {np.mean(results)}")
    print(results)
    print(np.mean(results))
    np.save("bayesian_HDL_results",results)
    # Save best_indices using pickle
    with open("bayesian_HDL_indices.pkl", "wb") as f:
        pickle.dump(best_indices_list, f)
    # Evaluate the black-box function for the initial set of indices
    initial_eval = black_box_function(initial_indices.flatten(), data, initial_points, attack_path, args, cfg)
    logging.critical(f"Initial Evaluation: {initial_eval}")
    
    # Reshape initial_evals to match the dimension required by `fit`
    initial_evals = np.full((initial_indices.shape[0],), initial_eval)
    
    # Fit the GP on the indices with their corresponding evaluations
    gp.fit(initial_indices, initial_evals)

    # Define bounds for optimization
    bounds = [(0, len(data) - 1) for _ in range(k)]

    # Perform the optimization
    def wrapped_acquisition(x):
        result = acquisition_function(x, gp, data, k)
        logging.critical(f"Evaluating acquisition function at {x}: {result}")
        return result
    
    result = minimize(
        wrapped_acquisition,
        x0=initial_indices.flatten(),
        bounds=bounds,
        method='L-BFGS-B'
    )

    logging.critical(f"Optimization Result for {attack_path}: {result}")
    if result.success:
        optimized_indices = np.clip(np.round(result.x).astype(int), 0, len(data) - 1)
        optimized_indices = np.unique(optimized_indices)[:k]  # Ensure we use exactly `k` unique indices
        optimized_eval = black_box_function(optimized_indices, data, initial_points, attack_path, args, cfg)
        logging.critical(f"Optimized Indices: {optimized_indices}")
        logging.critical(f"Optimized Evaluation: {optimized_eval}")
        logging.critical(f"Optimal Length Scale: {gp.kernel_.length_scale}")
        return optimized_eval
    else:
        logging.critical(f"Optimization failed: {result.message}")
        return 1e6

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    dataset, initial_points, attack_paths = load_attack_points_from_path(args, cfg)
    results = bayesian_optimisation_case(args, cfg, dataset[0], initial_points[0], attack_paths[0])
    logging.critical(f"Final Result: {results}")
    logging.critical(f"Mean: {np.mean(results)}")
    print(results)
    print(np.mean(results))