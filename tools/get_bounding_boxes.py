from cmath import inf
import open3d as o3d
import numpy as np
import torch

def roi_filter(points, roi_min=(0, -35, -35), roi_max=(35, 35, 35)):
    """
    Filter points within the specified region of interest (ROI).
    
    Args:
        points (np.ndarray): Input point cloud.
        roi_min (tuple): Minimum bounds of the ROI.
        roi_max (tuple): Maximum bounds of the ROI.
    
    Returns:
        o3d.geometry.PointCloud: Point cloud within the ROI.
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
    if bounding_box[3] < 4 and bounding_box[4] < 2.5:
        bounding_box[0] += (4 - bounding_box[3]) / 2
        bounding_box[3] = 4
    elif bounding_box[3] > 4 and bounding_box[3] < 1.6:
        bounding_box[0] += (1.6 - bounding_box[3]) / 2
        bounding_box[3] = 1.6
    
    difference = -1 * bounding_box[2] + bounding_box[5] / 2 - 2.2
    bounding_box[2] += difference / 2
    bounding_box[5] -= difference

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

def get_bounding_box(file):
    """
    Extract the bounding box for vehicles from a point cloud file.
    
    Args:
        file (str): Path to the point cloud file.
    
    Returns:
        np.ndarray: Bounding box of the vehicle points.
    """
    pc_array = np.load(file)
    vehicle_points = pc_array[pc_array[:, 3] == 6.0]
    vehicle_points = vehicle_points[:, :-1]
    if len(vehicle_points) < 4:
        return np.array([0, 0, 0, 0, 0, 0, 0])
    points = roi_filter(vehicle_points)
    return obb_to_bounding_box_format(points)

def center_to_corner2d(center, dim):
    """
    Convert center and dimensions to 2D corner points.
    
    Args:
        center (torch.Tensor): Center points.
        dim (torch.Tensor): Dimensions.
    
    Returns:
        torch.Tensor: Corner points.
    """
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], device=dim.device).type_as(center)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners

def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    """
    Calculate the intersection over union (IOU) for 3D bounding boxes.
    
    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes (N, 7).
        gt_boxes (torch.Tensor): Ground truth bounding boxes (N, 7).
    
    Returns:
        torch.Tensor: Maximum IOU values.
    """
    assert pred_boxes.shape[0] == gt_boxes.shape[0]
    
    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    return torch.max(volume_inter / volume_union)

def bbox3d_overlaps_diou_index(pred_boxes, gt_boxes):
    """
    Calculate the index of the bounding box with the highest IOU.
    
    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes (N, 7).
        gt_boxes (torch.Tensor): Ground truth bounding boxes (N, 7).
    
    Returns:
        int: Index of the bounding box with the highest IOU.
    """

    
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    return torch.argmax(volume_inter / volume_union).item()

def bbox3d_best_iou_bbox(pred_boxes, gt_boxes):
    """
    Get the bounding box with the highest IOU.
    
    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes (N, 7).
        gt_boxes (torch.Tensor): Ground truth bounding boxes (N, 7).
    
    Returns:
        torch.Tensor: Bounding box with the highest IOU.
    """
    
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    return pred_boxes[torch.argmax(volume_inter / volume_union)]
