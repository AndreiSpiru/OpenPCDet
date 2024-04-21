import validation
import validation_utils as utils
import os

#python evaluate_all_cases.py --cfg_file cfgs/kitti_models/pointpillar.yaml     --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10 --result_path slbz.xlsx

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    root_path = args.data_path
    for sensor_type in os.listdir(root_path):
        sensor_path = os.path.join(root_path, sensor_type)
        if os.path.isdir(sensor_path):
            for condition in os.listdir(sensor_path):
                condition_path = os.path.join(sensor_path, condition)
                if os.path.isdir(condition_path):
                    case_args = args
                    case_args.data_path = condition_path
                    iou, validation_file_list = validation.detection_iou(case_args, cfg)
                    for threshold in thresholds:
                        utils.create_or_modify_excel(case_args.result_path, case_args.data_path, threshold, case_args.ckpt, iou)