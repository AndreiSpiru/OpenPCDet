import validation
import validation_utils as utils
import os
from copy import copy
import random_ORA as ORA

#python evaluate_all_cases_per_budget.py --cfg_file cfgs/kitti_models/pointpillar.yaml     --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10 --result_path slbz.xlsx

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    budgets = [0, 10, 40, 60, 100, 200]
    for budget in budgets:
        case_args = copy(args)
        case_args.budget = budget
        ORA.random_ORA(case_args, cfg)
        root_path = args.data_path.replace("0-10", "0-10_attacked_random")
        for sensor_type in os.listdir(root_path):
            sensor_path = os.path.join(root_path, sensor_type)
            if os.path.isdir(sensor_path):
                for condition in os.listdir(sensor_path):
                    condition_path = os.path.join(sensor_path, condition)
                    if os.path.isdir(condition_path):
                        case_args.data_path = condition_path
                        iou, validation_file_list = validation.detection_iou(case_args, cfg)
                        for threshold in thresholds:
                            utils.create_or_modify_excel_recall(case_args.result_path, case_args.data_path, threshold, case_args.ckpt, iou, budget)