import validation
import validation_utils as utils
import os
import numpy as np

#python get_mean_confidence.py --cfg_file cfgs/kitti_models/pointpillar.yaml     --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    root_path = args.data_path
    confidences = []
    for condition in os.listdir(root_path):
        condition_path = os.path.join(root_path, condition)
        if os.path.isdir(condition_path):
            case_args = args
            case_args.data_path = condition_path
            confidence, validation_file_list = validation.detection_confidence(case_args, cfg)
            confidences.append(confidence)
    
    np_ious = np.array(flatten(confidences))
    print(np.mean(np_ious))
