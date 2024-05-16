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

#python bayesian_ORA.py --cfg_file cfgs/kitti_models/pointpillar.yaml    --budget 200 --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E/clear/1_labeled.bin

logging.basicConfig(filename='bayesian_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Simulated black-box function: Sum of selected points' intensities
def black_box_function(indices, data, initial_points, attack_path, args, cfg):
    indices = np.clip(np.round(indices).astype(int), 0, len(data) - 1)
    selected_points = data[indices]

    points = utils.shift_selected_points(initial_points, selected_points, 2)

    os.makedirs(os.path.dirname(attack_path), exist_ok=True)
    try:
        points.astype(np.float32).tofile(attack_path)
    except Exception as e:
        print(f"Failed to write to {attack_path}: {e}")

    scores, _ = validation.detection_iou_custom_dataset(args, cfg, [attack_path])
    return scores[0] # Example function: Sum the intensity values (assuming intensity is at index 3)

# Acquisition function to guide the Bayesian Optimization
def acquisition_function(x, model, data, n_select):
    x = np.clip(np.round(x).astype(int), 0, len(data) - 1)
    x = np.unique(x)  # Ensure indices are unique
    if len(x) < n_select:
        return 1e6  # Penalize for insufficient indices
    x = x.reshape(-1, 1)
    mu, sigma = model.predict(x, return_std=True)
    return -(mu - 1.96 * sigma).sum()  # Minimize the negative sum of LCB

# def load_attack_points_from_path(args, cfg):
#     bboxes, source_file_list = validation.detection_bboxes(args, cfg)
#     #print(bboxes)

#     file_npy = source_file_list[0][: -3] + "npy"
#     initial_points = np.load(file_npy)

#     bbox = torch.unsqueeze(bboxes[0], 0)
#     bbox = bbox.cpu().numpy()  

#     points, points_in_bbox, _ = utils.get_point_mask_in_boxes3d(initial_points, bbox)
#     non_zero_indices = points_in_bbox.squeeze().nonzero().squeeze()
    
#     attack_path = file_npy[: -4] + "bayesian.bin"

#     return non_zero_indices.numpy(), points.numpy(), attack_path

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
def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to delete file '{file_path}'.")

def bayesian_optimisation_case(args, cfg, data, initial_points, attack_path) :

        # Custom length scale bounds
    length_scale_bounds = (1e-2, 1e5)  # Adjust the upper bound to a larger value

    # Initialize Gaussian Process with custom bounds
    kernel = Matern(length_scale=1.0, length_scale_bounds=length_scale_bounds)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, n_restarts_optimizer=10)

    
    budget = min(args.budget, len(data))
    # Initial random selection and evaluation for each index
    initial_indices = np.random.choice(len(data), budget, replace=False)
    initial_evals = np.array([black_box_function([idx], data, initial_points, attack_path, args, cfg) for idx in initial_indices])

    # Fit the GP on the indices with their corresponding evaluations
    gp.fit(initial_indices.reshape(-1, 1), initial_evals)

    # Define bounds for optimization
    bounds = [(0, len(data) - 1) for _ in range(budget)]

    # Perform the optimization
    result = minimize(
        lambda x: acquisition_function(x, gp, data, budget),
        x0=initial_indices,
        bounds=bounds,
        method='L-BFGS-B'
    )

    # Output the optimization result
    logging.critical(f"Optimization Result for {attack_path}:{result}")
    if result.success:
        optimized_indices = np.clip(np.round(result.x).astype(int), 0, len(data) - 1)
        optimized_eval = black_box_function(optimized_indices, data, initial_points, attack_path, args, cfg)
        logging.critical(f"Optimized Indices:{optimized_indices}")
        logging.critical(f"Optimized Evaluation:{optimized_eval}")
        print("Optimized Evaluation:", optimized_eval)
        return optimized_eval
    else:
        logging.critical("Optimization failed:", result.message)
        return 1e6
    
    

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    dataset, initial_points, attack_paths = load_attack_points_from_path(args, cfg)
    results = [bayesian_optimisation_case(args, cfg, data, initial_point, attack_path) for (data, initial_point, attack_path) in zip (dataset, initial_points, attack_paths)]
    logging.critical(f"Final Result:{results}")
    logging.critical(f"Mean:{np.mean(results)}")
    print(results)
    print(np.mean(results))
    
