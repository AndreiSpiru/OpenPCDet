import argparse
import logging
import glob
import re
from pathlib import Path
import get_bounding_boxes as bbox
import os
import open3d as o3d

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from sklearn.decomposition import PCA


# python visualisation_test.py --cfg_file cfgs/kitti_models/pointpillar.yaml     --ckpt pointpillar_7728.pth     --data_path ~/mavs_code/output_data_converted/0-10/HDL-64E/clear/ 

def custom_sort_key(filepath):
    # Extract file name from file path
    filename = os.path.basename(filepath)
    # Extract numeric part using regular expression
    match = re.match(r'(\d+)_labeled\.bin', filename)
    if match:
        return int(match.group(1))
    else:
        return filename
    
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list = sorted(data_file_list)
        #print(data_file_list)
        validation_file_list = glob.glob(str(root_path / f'*verification.npy')) if self.root_path.is_dir() else [self.root_path]
        validation_file_list = sorted(validation_file_list)
        #print(validation_file_list)
        self.validation_file_list = validation_file_list
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--result_path', type=str, default='evaluation_results.xlsx',
                        help='specify the location for the saved results')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--budget', type=int, default='0', help='specify budget of ORA attack')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def normalize_heading(heading):
    """Normalize heading to the range [0, 2π)."""
    while heading < 0:
        heading += 2 * np.pi
    while heading >= 2 * np.pi:
        heading -= 2 * np.pi
    return heading

def roi_filter1(points, roi_min=(0,-35,-35), roi_max=(35,35,35)):

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

    return roi_points

def roi_filter(points, roi_min=(0,-35,-35), roi_max=(35,35,35)):

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

    # Create a new point cloud with the filtered points
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    return roi_pcd

def obb_to_bounding_box_format(roi_pcd):
    obb = roi_pcd.get_oriented_bounding_box()
    center = obb.center
    extents = obb.extent
    rotation_matrix = obb.R
    
    # Extract the heading from the rotation matrix
    heading_vector = rotation_matrix[:, 0]  # First column
    heading = np.arctan2(heading_vector[1], heading_vector[0])
    
    # Normalize heading to the range [0, 2π)
    if heading < 0:
        heading += 2 * np.pi
    
    # Create the bounding box in the desired format
    bounding_box = np.concatenate([center, extents, [heading]])
    return bounding_box

def get_obb(file):
    pc_array = np.load(file)
    vehicle_points = pc_array[pc_array[:,3] == 6.0]
    vehicle_points = vehicle_points[:, :-1]
    points = roi_filter(vehicle_points)
    return obb_to_bounding_box_format(points)

def get_bounding_box1(file):
    pc_array = np.load(file)
    vehicle_points = pc_array[pc_array[:,3] == 6.0]
    vehicle_points = vehicle_points[:, :-1]
    points = roi_filter1(vehicle_points)
    print(points)
    # Calculate the bounding box center
    center = np.mean(points, axis=0)
    
    # Calculate the dimensions of the bounding box
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    dimensions = max_point - min_point
    
    # Perform PCA to find the orientation
    pca = PCA(n_components=3)
    pca.fit(points)
    
   # The heading is the angle of the first principal component in the xy-plane
    heading_vector = pca.components_[0]
    print(f"PCA first component: {heading_vector}")
    heading = np.arctan2(heading_vector[1], heading_vector[0])
    print(f"Raw heading in radians: {heading}")
    
    # Normalize heading to the range [0, 2π)
    heading = normalize_heading(heading)
    print(f"Normalized heading in radians: {heading}")
    
    print(np.concatenate([center, dimensions, [heading]]))
    return np.concatenate([center, dimensions, [heading]])

def _bboxes_to_corners2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                dtype=torch.float32, device=dim.device)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners

def bbox3d_overlaps_iou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = _bboxes_to_corners2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = _bboxes_to_corners2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
   
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter
    print(inter)
    print(inter_h)
    print(volume_inter)
    print(volume_union)
    ious = volume_inter / volume_union
    ious = torch.clamp(ious, min=0, max=1.0)
    return torch.max(ious)


def detection_bboxes(args, cfg):
    
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
    validation_bboxes1 = [get_obb(file) for file in validation_file_list]
    # print(validation_bboxes)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    total = 0
    correct = 0
    bboxes = []
    iou = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):  
            total += 1
            data_dict = demo_dataset.collate_batch([data_dict])
            #print(data_dict)
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            #print(pred_dicts)
            mask = pred_dicts[0]['pred_labels'] == 1
            vehicle_bboxes = pred_dicts[0]['pred_boxes'][mask]
            if(vehicle_bboxes.numel() == 0):
                bboxes.append(torch.tensor(-1))
            else:
                true_bbox1 = validation_bboxes1[idx]

                true_bbox1 = np.tile(true_bbox1, (vehicle_bboxes.shape[0],1))
                true_bbox1 = torch.tensor(true_bbox1, device ='cuda')

               
                max = bbox3d_overlaps_iou(true_bbox1, vehicle_bboxes)
                max = max.cpu()
                iou.append(max)
                print(max)
                print(vehicle_bboxes)
                print(true_bbox1)
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=true_bbox1,
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    #print(list(zip(validation_file_names, bboxes)))
    print(iou)
    logger.info('Demo done.')
    return iou

def flatten(lst):
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