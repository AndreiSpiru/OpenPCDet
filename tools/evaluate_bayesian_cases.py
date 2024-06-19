import validation
import validation_utils as utils
import os
from copy import copy
import random_ORA as ORA
import numpy as np

# Command to run the script:
# python evaluate_bayesian_cases.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E --result_path bayesian_separate_results.xlsx

def main():
    """
    Main function to evaluate all cases for Object Removal Attacks (ORA) per budget and save results to an Excel file.
    Used to convert the outputed results for a separate BORA to usable excel format
    """
    args, cfg = validation.parse_config()  # Parse configuration and arguments
    root_path = args.data_path
    # Path to your pickle file
    result_file_path = 'bayesian_results/new/HDL/SECOND/bayesian_HDL_results.npy'
    attack_paths = []
    results = np.load(result_file_path)
    for condition in os.listdir(root_path):
        condition_path = os.path.join(root_path, condition)

        if os.path.isdir(condition_path):
            case_args = copy(args)
            case_args.data_path = condition_path
            bboxes, source_file_list = validation.detection_bboxes(case_args, cfg)
            for idx, file_bin in enumerate(source_file_list):
                file_npy = file_bin[:-3] + "npy"
                base_file_npy = os.path.basename(file_npy)
                attack_path = os.path.join(condition_path, base_file_npy)
                print(attack_path)
                attack_paths.append(attack_path)

    utils.create_or_modify_excel_generic(results, attack_paths, args.ckpt, type="bayesian_separate", file_path=args.result_path)    

if __name__ == '__main__':
    main()
