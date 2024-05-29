import validation
import validation_utils as utils
import os
from copy import copy
import random_ORA as ORA

# Command to run the script:
# python evaluate_all_cases_per_budget.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10 --result_path slbz.xlsx

def main():
    """
    Main function to evaluate all cases for Object Removal Attacks (ORA) per budget and save results to an Excel file.
    This file is mainly use to create data for comparison with the original ORA paper.
    """
    args, cfg = validation.parse_config()  # Parse configuration and arguments
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # IOU thresholds for evaluation
    budgets = [0, 10, 40, 60, 100, 200]  # Different budgets for the attacks

    for budget in budgets:
        case_args = copy(args)  # Create a copy of arguments for each budget
        case_args.budget = budget

        # Apply random Object Removal Attack (ORA) with the specified budget
        ORA.random_ORA(case_args, cfg)

        # Update root path to reflect the attacked data
        root_path = args.data_path.replace("0-10", "0-10_attacked_random")

        for sensor_type in os.listdir(root_path):
            sensor_path = os.path.join(root_path, sensor_type)

            if os.path.isdir(sensor_path):
                for condition in os.listdir(sensor_path):
                    condition_path = os.path.join(sensor_path, condition)

                    if os.path.isdir(condition_path):
                        case_args.data_path = condition_path

                        attack_paths = utils.get_scenarios_in_distance_interval(condition_path, 0, 20)
                        # Calculate IOU and get the validation file list
                        iou, _ = validation.detection_iou_custom_dataset(case_args, cfg, attack_paths)

                        for threshold in thresholds:
                            # Create or modify the Excel file with the evaluation results
                            utils.create_or_modify_excel_recall(case_args.result_path, case_args.data_path, threshold, case_args.ckpt, iou, budget)

if __name__ == '__main__':
    main()
