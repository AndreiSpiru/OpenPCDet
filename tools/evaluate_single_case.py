import validation
import validation_utils as utils

# Command to run the script:
# python evaluate_single_case.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E/clear/ --result_path slbz.xlsx

def main():
    """
    Main function to validate the dataset using the specified configuration and save results to an Excel file.
    """
    args, cfg = validation.parse_config()  # Parse configuration and arguments
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # IOU thresholds for evaluation

    # Calculate IOU and get the validation file list
    iou, _ = validation.detection_iou(args, cfg)

    for threshold in thresholds:
        # Create or modify the Excel file with the evaluation results
        utils.create_or_modify_excel(args.result_path, args.data_path, threshold, args.ckpt, iou)

if __name__ == '__main__':
    main()
