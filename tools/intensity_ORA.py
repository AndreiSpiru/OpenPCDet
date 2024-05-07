import validation
import validation_utils as val_utils
import ORA_utils as utils
import os
import numpy as np
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
import torch

#python intensity_ORA.py --cfg_file cfgs/kitti_models/pointpillar.yaml    --budget 200 --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/


if __name__ == '__main__':
    args, cfg = validation.parse_config()
    root_path = args.data_path
    root_path_attack = os.path.dirname(root_path.rstrip("/"))
    root_path_attack = os.path.join(root_path_attack, "0-10_attacked_intensity")
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
                            updated_points = utils.apply_intensity_ORA_points_in_boxes3d(initial_points, bbox, args.budget)
                            
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
