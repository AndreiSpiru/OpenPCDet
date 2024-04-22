import validation
import validation_utils as utils
import os
import numpy as np
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
import torch

#python random_ORA.py --cfg_file cfgs/kitti_models/pointpillar.yaml    --budget 200 --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/

def remove_points_in_boxes3d(points, boxes3d, budget):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)

    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    #print(point_masks)

    # Get the indices of non-zero values
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    # Shuffle the indices
    shuffled_indices = torch.randperm(non_zero_indices.numel())

    # Select the first k shuffled indices
    k = min(budget, shuffled_indices.numel())  # Choose the number of non-zero values to keep
    selected_indices = shuffled_indices[:k]

    # Create a mask to set non-selected indices to zero
    mask = torch.zeros_like(non_zero_indices, dtype=torch.bool)
    mask[selected_indices] = True

    # Set non-selected non-zero values to zero
    point_masks[:, non_zero_indices[mask == False]] = 0

    points = points[point_masks.sum(dim=0) == 0]    
    return points.numpy() if is_numpy else points

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    root_path = args.data_path
    root_path_attack = os.path.dirname(root_path.rstrip("/"))
    root_path_attack = os.path.join(root_path_attack, "0-10_attacked")
    TMP_ok = False
    for sensor_type in os.listdir(root_path):
        sensor_path = os.path.join(root_path, sensor_type)
        sensor_path_attack = os.path.join(root_path_attack, sensor_type)
        if os.path.isdir(sensor_path):
            for condition in os.listdir(sensor_path):
                condition_path = os.path.join(sensor_path, condition)
                condition_path_attack = os.path.join(sensor_path_attack, condition)
                if os.path.isdir(condition_path):
                    case_args = args
                    case_args.data_path = condition_path
                    bboxes, source_file_list = validation.detection_bboxes(case_args, cfg)
                    #print(bboxes)
                    for idx, file_bin in enumerate(source_file_list):
                        #if TMP_ok == False:
                            file_npy = file_bin[: -3] + "npy"
                            initial_points = np.load(file_npy)
                            bbox = torch.unsqueeze(bboxes[idx], 0)
                            bbox = bbox.cpu().numpy()  
                            #print(bbox)
                            updated_points = remove_points_in_boxes3d(initial_points, bbox, args.budget)
                            
                            # # Convert arrays to sets of tuples
                            # set1 = set(map(tuple, initial_points))
                            # set2 = set(map(tuple, updated_points))

                            # # Get the rows that exist in array1 but not in array2
                            # unique_rows = np.array([row for row in set1 if row not in set2])
                            # print(unique_rows.shape)

                            base_file_npy = os.path.basename(file_npy)
                            attack_path = os.path.join(condition_path_attack, base_file_npy)
                            os.makedirs(os.path.dirname(attack_path), exist_ok=True)
                            np.save(attack_path, updated_points)

                            base_file_bin = base_file_npy[:-3] + "bin"
                            attack_path_bin = os.path.join(condition_path_attack, base_file_bin)
                            updated_points.astype(np.float32).tofile(attack_path_bin)
                            TMP_ok = True
