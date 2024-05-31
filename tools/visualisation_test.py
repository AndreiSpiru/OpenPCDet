import argparse
import logging
import glob
import re
from pathlib import Path
import get_bounding_boxes as bbox
import os
import numpy as np
import torch
import open3d as o3d
from sklearn.decomposition import PCA

try:
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except ImportError:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# Command to run:
# python3 visualisation_test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt pointpillar_7728.pth --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E/clear/

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

def normalize_heading(heading):
    """
    Normalize heading to the range [0, 2π).
    
    Args:
        heading (float): Heading angle in radians.
        
    Returns:
        float: Normalized heading angle.
    """
    while heading < 0:
        heading += 2 * np.pi
    while heading >= 2 * np.pi:
        heading -= 2 * np.pi
    return heading

def roi_filter(points, roi_min=(0, -35, -35), roi_max=(35, 35, 35)):
    """
    Filter points based on Region of Interest (ROI).
    
    Args:
        points (ndarray): Point cloud data.
        roi_min (tuple): Minimum ROI boundary.
        roi_max (tuple): Maximum ROI boundary.
        
    Returns:
        o3d.geometry.PointCloud: Filtered points within the ROI.
    """
    mask_roi = np.logical_and.reduce((
        points[:, 0] >= roi_min[0],
        points[:, 0] <= roi_max[0],
        abs(points[:, 1] / points[:, 0]) <= 5,
        points[:, 1] >= roi_min[1],
        points[:, 1] <= roi_max[1],
        points[:, 2] >= roi_min[2],
        points[:, 2] <= roi_max[2]
    ))

    roi_points = points[mask_roi]
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    return roi_pcd

def augment_dimensions(bounding_box, is_flipped):
    if not is_flipped: 
        if  bounding_box[3] < 4 and bounding_box[4] < bounding_box[3]:
            bounding_box[0] += (4 - bounding_box[3]) / 2
            bounding_box[3] = 4
        elif bounding_box[3] < 1.6 and bounding_box[4] > bounding_box[3] :
            bounding_box[0] += (1.6 - bounding_box[3]) / 2
            bounding_box[3] = 1.6
    
    elif is_flipped:
        if bounding_box[4] < 4 and bounding_box[3] < bounding_box[4]:
            bounding_box[0] += (4 - bounding_box[4]) / 2
            bounding_box[4] = 4
        if bounding_box[4] < 1.6 and bounding_box[3] > bounding_box[4] :
            bounding_box[0] += (1.6 - bounding_box[4]) / 2
            bounding_box[4] = 1.6
    
    # print(-1 * bounding_box[2] + bounding_box[5] / 2)
    # if (-1 * bounding_box[2] + bounding_box[5] / 2) > 2.6:
    difference = -1 * bounding_box[2] + bounding_box[5] / 2 - 2.2
    print(difference)
    bounding_box[2] += difference / 2
    bounding_box[5] -= difference
    # elif (-1 * bounding_box[2] + bounding_box[5] / 2) < 2.6:
    #     difference = -1 * bounding_box[2] + bounding_box[5] / 2 - 2.6
    #     print(difference)
    #     bounding_box[2] -= difference / 2
    #     bounding_box[5] += difference
    
    print(bounding_box)
    print(is_flipped)
    return bounding_box



def obb_to_bounding_box_format(roi_pcd):
    """
    Convert Oriented Bounding Box (OBB) to bounding box format.
    
    Args:
        roi_pcd (o3d.geometry.PointCloud): Point cloud data within ROI.
        
    Returns:
        tuple: Bounding box parameters [center, extents, heading], 
               and boolean indicating if extents are flipped.
    """
    obb = roi_pcd.get_oriented_bounding_box()
    center = obb.center
    extents = obb.extent
    rotation_matrix = obb.R
    
    # Extract the heading from the rotation matrix
    heading_vector = rotation_matrix[:, 0]  # First column
    heading = np.arctan2(heading_vector[1], heading_vector[0])
    
    if heading < 0:
        heading += 2 * np.pi

    # Identify the correspondence of extents to global axes
    local_axes = rotation_matrix.T  # Transpose to get local to global
    abs_axes = np.abs(local_axes)
    
    # Determine the primary direction of each local axis
    major_directions = np.argmax(abs_axes, axis=1)
    
    # Check if the first and second extents are flipped
    is_flipped = (major_directions[0] == 1 and major_directions[1] == 0)
    
    bounding_box = np.concatenate([center, extents, [heading]])
    bounding_box = augment_dimensions(bounding_box, is_flipped)
    
    return bounding_box

def obb_to_bounding_box_format1(roi_pcd):
    """
    Convert Oriented Bounding Box (OBB) to bounding box format.
    
    Args:
        roi_pcd (o3d.geometry.PointCloud): Point cloud data within ROI.
        
    Returns:
        tuple: Bounding box parameters [center, extents, heading], 
               and boolean indicating if extents are flipped.
    """
    obb = roi_pcd.get_oriented_bounding_box()
    center = obb.center
    extents = obb.extent
    rotation_matrix = obb.R
    
    # Extract the heading from the rotation matrix
    heading_vector = rotation_matrix[:, 0]  # First column
    heading = np.arctan2(heading_vector[1], heading_vector[0])
    
    if heading < 0:
        heading += 2 * np.pi

    # Identify the correspondence of extents to global axes
    local_axes = rotation_matrix.T  # Transpose to get local to global
    abs_axes = np.abs(local_axes)
    
    # Determine the primary direction of each local axis
    major_directions = np.argmax(abs_axes, axis=1)
    
    # Check if the first and second extents are flipped
    is_flipped = (major_directions[0] == 1 and major_directions[1] == 0)
    
    bounding_box = np.concatenate([center, extents, [heading]])
    
    return bounding_box

def get_obb(file):
    """
    Get Oriented Bounding Box (OBB) for a given file.
    
    Args:
        file (str): Path to the point cloud file.
        
    Returns:
        ndarray: Bounding box parameters [center, extents, heading].
    """
    pc_array = np.load(file)
    vehicle_points = pc_array[pc_array[:, 3] == 6.0]
    vehicle_points = vehicle_points[:, :-1]
    points = roi_filter(vehicle_points)
    return obb_to_bounding_box_format(points)

def get_obb1(file):
    """
    Get Oriented Bounding Box (OBB) for a given file.
    
    Args:
        file (str): Path to the point cloud file.
        
    Returns:
        ndarray: Bounding box parameters [center, extents, heading].
    """
    pc_array = np.load(file)
    vehicle_points = pc_array[pc_array[:, 3] == 6.0]
    vehicle_points = vehicle_points[:, :-1]
    points = roi_filter(vehicle_points)
    return obb_to_bounding_box_format1(points)

def _bboxes_to_corners2d(center, dim):
    """
    Convert bounding boxes to 2D corners.
    
    Args:
        center (tensor): Bounding box center coordinates.
        dim (tensor): Bounding box dimensions.
        
    Returns:
        tensor: 2D corners of bounding boxes.
    """
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=torch.float32, device=dim.device)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners

def bbox3d_overlaps_iou(pred_boxes, gt_boxes):
    """
    Compute 3D IoU between predicted and ground truth bounding boxes.
    
    Args:
        pred_boxes (tensor): Predicted bounding boxes.
        gt_boxes (tensor): Ground truth bounding boxes.
        
    Returns:
        tensor: Maximum IoU values.
    """
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = _bboxes_to_corners2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = _bboxes_to_corners2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
   
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter
    ious = volume_inter / volume_union

    print(volume_inter)
    print(volume_union)
    ious = torch.clamp(ious, min=0, max=1.0)
    return torch.max(ious)

def detection_bboxes(args, cfg):
    """
    Perform detection and return bounding boxes for the dataset.
    
    Args:
        args: Command line arguments.
        cfg: Configuration object.
        
    Returns:
        list: IoU values.
    """
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    validation_file_list = demo_dataset.validation_file_list
    
    validation_bboxes = [get_obb1(file) for file in validation_file_list]
    validation_bboxes1 = [get_obb(file) for file in validation_file_list]
    
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
                print(pred_dicts[0]['pred_boxes'][0])
                true_bbox = validation_bboxes[idx]
                print(true_bbox)
                true_bbox = np.tile(true_bbox, (vehicle_bboxes.shape[0], 1))
                true_bbox = torch.tensor(true_bbox, device='cuda')

                true_bbox1 = validation_bboxes1[idx]
                print(true_bbox1)
                true_bbox1 = np.tile(true_bbox1, (vehicle_bboxes.shape[0], 1))
                true_bbox1 = torch.tensor(true_bbox1, device='cuda')
                
                max_iou = bbox3d_overlaps_iou(true_bbox, vehicle_bboxes)
                max_iou1 = bbox3d_overlaps_iou(true_bbox1, vehicle_bboxes)
                print(max_iou)
                print(max_iou1)
                max_iou1 = max_iou1.cpu()
                iou.append(max_iou1.item())
                
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                # )
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=true_bbox,
                #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                # )
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=true_bbox1,
                #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                # )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    
    logger.info('Demo done.')
    print(iou)
    return iou

def flatten(lst):
    """
    Flatten a nested list.
    
    Args:
        lst (list): Nested list.
        
    Returns:
        list: Flattened list.
    """
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

if __name__ == '__main__':
    args, cfg = parse_config()
    detection_bboxes(args, cfg)
