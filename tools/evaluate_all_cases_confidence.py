import validation
import validation_utils as utils
import os

# Command to run the script:
# python evaluate_all_cases_confidence.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10 --result_path slbz.xlsx

def main():
    """
    Main function to evaluate all cases for Object Removal Attacks (ORA) and save results to an Excel file.
    """
    # Parse configuration and arguments
    args, cfg = validation.parse_config()
    
    # Confidence thresholds for evaluation
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    root_path = args.data_path

    # Iterate over sensor types in the root path
    for sensor_type in os.listdir(root_path):
        sensor_path = os.path.join(root_path, sensor_type)

        if os.path.isdir(sensor_path):
            # Iterate over conditions in the sensor path
            for condition in os.listdir(sensor_path):
                condition_path = os.path.join(sensor_path, condition)

                if os.path.isdir(condition_path):
                    case_args = args
                    case_args.data_path = condition_path

                    # Calculate confidence and get the validation file list
                    confidence, validation_file_list = validation.detection_confidence(case_args, cfg)
                    print(confidence)

                    # Evaluate results for each confidence threshold
                    for threshold in thresholds:
                        # Create or modify the Excel file with the evaluation results
                        utils.create_or_modify_excel_recall_confidence(
                            case_args.result_path,
                            case_args.data_path,
                            threshold,
                            case_args.ckpt,
                            confidence,
                            200,
                            "distance_positive"   # should be modified to the type of attack evaluated
                        )

if __name__ == '__main__':
    main()

