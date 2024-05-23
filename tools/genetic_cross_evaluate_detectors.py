import validation
import ORA_utils as utils
import os
from copy import copy
import random_ORA as ORA
import pickle
import torch
import numpy as np

# Command to run the script:
# python genetic_cross_evaluate_detectors.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

def flatten(lst):
    """
    Flatten a nested list.

    Args:
        lst (list): Nested list to flatten.

    Returns:
        list: Flattened list.
    """
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def main():
    """
    Main function to apply Object Removal Attacks (ORA) using pre-selected points
    and evaluate detectors' performance.
    Purpose of the script is to evaluate the performance of a pretrained genetic model
    on different object detectors.
    """
    args, cfg = validation.parse_config()  # Parse configuration and arguments
    saved_indices = "genetic_algorithm_results/HDL-64E/PointPillar/200/best_individual.pkl"
    
    # Load the pre-selected indices from a pickle file
    with open(saved_indices, 'rb') as f:
        selected_indices = pickle.load(f)

    print(selected_indices)
    max_length = 5971
    root_path = args.data_path
    ious = []

    for condition in os.listdir(root_path):
        condition_path = os.path.join(root_path, condition)

        if os.path.isdir(condition_path):
            case_args = copy(args)
            case_args.data_path = condition_path
            print("Processing:", case_args.data_path)
            
            # Get bounding boxes and source file list
            bboxes, source_file_list = validation.detection_bboxes(case_args, cfg)

            for idx, file_bin in enumerate(source_file_list):
                file_npy = file_bin[:-3] + "npy"
                initial_points = np.load(file_npy)

                bbox = torch.unsqueeze(bboxes[idx], 0)
                bbox = bbox.cpu().numpy()

                # Apply ORA using pre-selected points
                attacked_points = utils.apply_ORA_pre_selected_points(initial_points, selected_indices, bbox, max_length)

                attack_path_bin = file_bin.replace("0-10", "0-10_genetic_cross_detector")
                os.makedirs(os.path.dirname(attack_path_bin), exist_ok=True)

                try:
                    attacked_points.astype(np.float32).tofile(attack_path_bin)
                except Exception as e:
                    print(f"Failed to write to {attack_path_bin}: {e}")

            case_args_iou = copy(args)
            case_args_iou.data_path = condition_path.replace("0-10", "0-10_genetic_cross_detector")
            print("Evaluating IOU for:", case_args_iou.data_path)

            # Calculate IOU and get the validation file list
            iou, _ = validation.detection_iou(case_args_iou, cfg)
            ious.append(iou)
    
    # Flatten the list of IOUs and calculate the mean
    np_ious = np.array(flatten(ious))
    print("Mean IOU:", np.mean(np_ious))

if __name__ == '__main__':
    main()
