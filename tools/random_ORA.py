import validation
import validation_utils as utils
import os

#python random_ORA.py --cfg_file cfgs/kitti_models/pointpillar.yaml    --budget 200 --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/

if __name__ == '__main__':
    args, cfg = validation.parse_config()
    root_path = args.data_path
    root_path_attack = os.path.dirname(root_path.rstrip("/"))
    root_path_attack = os.path.join(root_path_attack, "0-10_attacked")
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
                    print(condition_path_attack)
                    bboxes, source_file_list = validation.detection_bboxes(case_args, cfg)