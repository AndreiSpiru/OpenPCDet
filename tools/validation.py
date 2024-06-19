import argparse
import logging
import glob
import re
from pathlib import Path
import get_bounding_boxes as bbox
import os
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# Import visualization libraries
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except ImportError:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

# Validation methods, does the integration between our attacks and OpenPCDet
# Based on the demo.py file

def custom_sort_key(filepath):
    """
    Custom sort key to extract numeric part from the filename.
    """
    filename = os.path.basename(filepath)
    match = re.match(r'(\d+)_labeled\.bin', filename)
    if match:
        return int(match.group(1))
    else:
        return filename

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Initialize DemoDataset.
        
        Args:
            dataset_cfg: Configuration for the dataset.
            class_names: List of class names.
            training: Boolean indicating training mode.
            root_path: Path to the root directory of the dataset.
            logger: Logger object.
            ext: File extension for the point cloud data.
        """
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        self.sample_file_list = sorted(data_file_list)
        validation_file_list = glob.glob(str(root_path / f'*verification.npy')) if self.root_path.is_dir() else [self.root_path]
        self.validation_file_list = sorted(validation_file_list)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError("Unsupported file extension")

        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, attack_paths, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Initialize CustomDataset.
        
        Args:
            dataset_cfg: Configuration for the dataset.
            class_names: List of class names.
            attack_paths: List of paths to attacked point cloud data.
            training: Boolean indicating training mode.
            root_path: Path to the root directory of the dataset.
            logger: Logger object.
            ext: File extension for the point cloud data.
        """
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext
        self.sample_file_list = [file_path[:-4] + ".bin" for file_path in attack_paths]
        self.validation_file_list = [file_path[:-4] + "_verification.npy" for file_path in attack_paths]

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError("Unsupported file extension")

        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    """
    Parse command line arguments and configuration file.
    
    Returns:
        args: Parsed command line arguments.
        cfg: Configuration object.
    """
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml', help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data', help='specify the point cloud data file or directory')
    parser.add_argument('--result_path', type=str, default='evaluation_results.xlsx', help='specify the location for the saved results')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--budget', type=int, default=0, help='specify budget of ORA attack')
    
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def detection_iou(args, cfg):
    """
    Perform detection and compute IoU for the dataset.
    
    Args:
        args: Command line arguments.
        cfg: Configuration object.
    
    Returns:
        iou: List of IoU values.
        validation_file_list: List of validation file paths.
    """
    logging.disable(logging.CRITICAL)
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    
    validation_file_list = demo_dataset.validation_file_list
    validation_bboxes = [bbox.get_bounding_box(file) for file in validation_file_list]
    validation_bboxes = np.stack(validation_bboxes)
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    iou = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            mask = pred_dicts[0]['pred_labels'] == 1
            vehicle_bboxes = pred_dicts[0]['pred_boxes'][mask]
            
            if vehicle_bboxes.numel() == 0:
                iou.append(0)
            else:
                true_bbox = validation_bboxes[idx]
                true_bbox = np.tile(true_bbox, (vehicle_bboxes.shape[0], 1))
                true_bbox = torch.tensor(true_bbox, device='cuda')
                
                max_iou = bbox.bbox3d_overlaps_diou(true_bbox, vehicle_bboxes)
                iou.append(max_iou.item())
            
            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    
    return iou, validation_file_list

def detection_confidence(args, cfg):
    """
    Perform detection and compute confidence for the dataset.
    
    Args:
        args: Command line arguments.
        cfg: Configuration object.
    
    Returns:
        confidences: List of confidence scores.
        validation_file_list: List of validation file paths.
    """
    logging.disable(logging.CRITICAL)
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    
    validation_file_list = demo_dataset.validation_file_list
    validation_bboxes = [bbox.get_bounding_box(file) for file in validation_file_list]
    validation_bboxes = np.stack(validation_bboxes)
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    confidences = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            mask = pred_dicts[0]['pred_labels'] == 1
            vehicle_bboxes = pred_dicts[0]['pred_boxes'][mask]
            
            if vehicle_bboxes.numel() == 0:
                confidences.append(0)
            else:
                true_bbox = validation_bboxes[idx]
                true_bbox = np.tile(true_bbox, (vehicle_bboxes.shape[0], 1))
                true_bbox = torch.tensor(true_bbox, device='cuda')
                
                max_idx = bbox.bbox3d_overlaps_diou_index(true_bbox, vehicle_bboxes)
                best_confidence = pred_dicts[0]['pred_scores'][max_idx]
                confidences.append(best_confidence.item())
            
            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    
    return confidences, validation_file_list

def detection_confidence_custom_dataset(args, cfg, attack_file_paths):
    """
    Perform detection and compute IoU for a custom dataset.
    
    Args:
        args: Command line arguments.
        cfg: Configuration object.
        attack_file_paths: List of paths to attacked point cloud data.
    
    Returns:
        iou: List of IoU values.
        validation_file_list: List of validation file paths.
    """
    logging.disable(logging.INFO)
    logger = common_utils.create_logger()
    demo_dataset = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, attack_paths=attack_file_paths, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    
    validation_file_list = demo_dataset.validation_file_list
    validation_bboxes = [bbox.get_bounding_box(file) for file in validation_file_list]
    validation_bboxes = np.stack(validation_bboxes)
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    confidences = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            mask = pred_dicts[0]['pred_labels'] == 1
            vehicle_bboxes = pred_dicts[0]['pred_boxes'][mask]
            
            if vehicle_bboxes.numel() == 0:
                confidences.append(0)
            else:
                true_bbox = validation_bboxes[idx]
                true_bbox = np.tile(true_bbox, (vehicle_bboxes.shape[0], 1))
                true_bbox = torch.tensor(true_bbox, device='cuda')
                
                max_idx = bbox.bbox3d_overlaps_diou_index(true_bbox, vehicle_bboxes)
                best_confidence = pred_dicts[0]['pred_scores'][max_idx]
                confidences.append(best_confidence.item())
            
            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    
    return confidences, validation_file_list

def detection_iou_custom_dataset(args, cfg, attack_file_paths):
    """
    Perform detection and compute IoU for a custom dataset.
    
    Args:
        args: Command line arguments.
        cfg: Configuration object.
        attack_file_paths: List of paths to attacked point cloud data.
    
    Returns:
        iou: List of IoU values.
        validation_file_list: List of validation file paths.
    """
    logging.disable(logging.INFO)
    logger = common_utils.create_logger()
    demo_dataset = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, attack_paths=attack_file_paths, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    
    validation_file_list = demo_dataset.validation_file_list
    validation_bboxes = [bbox.get_bounding_box(file) for file in validation_file_list]
    validation_bboxes = np.stack(validation_bboxes)
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    iou = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            mask = pred_dicts[0]['pred_labels'] == 1
            vehicle_bboxes = pred_dicts[0]['pred_boxes'][mask]
            
            if vehicle_bboxes.numel() == 0:
                iou.append(0)
            else:
                true_bbox = validation_bboxes[idx]
                true_bbox = np.tile(true_bbox, (vehicle_bboxes.shape[0], 1))
                true_bbox = torch.tensor(true_bbox, device='cuda')
                
                max_iou = bbox.bbox3d_overlaps_diou(true_bbox, vehicle_bboxes)
                iou.append(max_iou.item())
            
            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    
    return iou, validation_file_list

def detection_bboxes(args, cfg):
    """
    Perform detection and return bounding boxes for the dataset.
    
    Args:
        args: Command line arguments.
        cfg: Configuration object.
    
    Returns:
        bboxes: List of bounding boxes.
        sample_file_list: List of sample file paths.
    """
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    validation_file_list = demo_dataset.validation_file_list
    validation_bboxes = [bbox.get_bounding_box(file) for file in validation_file_list]
    validation_bboxes = np.stack(validation_bboxes)
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    bboxes = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            mask = pred_dicts[0]['pred_labels'] == 1
            vehicle_bboxes = pred_dicts[0]['pred_boxes'][mask]
            
            if vehicle_bboxes.numel() == 0:
                bboxes.append(torch.tensor([-1, -1, -1, -1, -1, -1, -1]))
            else:
                true_bbox = validation_bboxes[idx]
                true_bbox = np.tile(true_bbox, (vehicle_bboxes.shape[0], 1))
                true_bbox = torch.tensor(true_bbox, device='cuda')
                
                best_bbox = bbox.bbox3d_best_iou_bbox(vehicle_bboxes, true_bbox)
                bboxes.append(best_bbox)
            
            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    
    return bboxes, demo_dataset.sample_file_list
