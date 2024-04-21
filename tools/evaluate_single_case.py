import validation
import validation_utils as utils

#python validation.py --cfg_file cfgs/kitti_models/pointpillar.yaml     --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E/clear/ --result_path slbz.xlsx

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    iou, validation_file_list = validation.detection_iou(args, cfg)
    for threshold in thresholds:
        utils.create_or_modify_excel(args.result_path, args.data_path, threshold, args.ckpt, iou)