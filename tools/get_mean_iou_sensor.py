import validation
import validation_utils as utils
import os
import numpy as np

# Command to run the script:
# python get_mean_iou_sensor.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

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
    Main function to calculate the mean IOU of detections in the dataset.
    """
    args, cfg = validation.parse_config()  # Parse configuration and arguments
    root_path = args.data_path
    ious = []

    for condition in os.listdir(root_path):
        condition_path = os.path.join(root_path, condition)

        if os.path.isdir(condition_path):
            case_args = args
            case_args.data_path = condition_path

            # Calculate IOU and get the validation file list
            iou, validation_file_list = validation.detection_iou(case_args, cfg)
            ious.append(iou)

    # Flatten the list of IOU scores and calculate the mean
    flattened_ious = np.array(flatten(ious))
    mean_iou = np.mean(flattened_ious)
    print("Mean IOU:", mean_iou)

if __name__ == '__main__':
    main()
