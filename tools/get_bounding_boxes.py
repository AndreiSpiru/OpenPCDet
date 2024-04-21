from cmath import inf
import open3d as o3d
import numpy as np
import torch

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

def center_to_corner2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], device=dim.device).type_as(center)  # (4, 2)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])  # (N, 4, 2)
    corners = corners + center.view(-1, 1, 2)
    return corners

def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    """
    https://github.com/agent-sgs/PillarNet/blob/master/det3d/core/utils/center_utils.py
    Args:
        pred_boxes (N, 7): 
        gt_boxes (N, 7): 

    Returns:
        _type_: _description_
    """
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])  # (N, 4, 2)
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])  # (N, 4, 2)   

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # print(volume_union)
    # print(volume_inter)
    print(volume_inter / volume_union)

    return torch.max(volume_inter / volume_union)

def bbox3d_best_iou_bbox(pred_boxes, gt_boxes):
    """
    https://github.com/agent-sgs/PillarNet/blob/master/det3d/core/utils/center_utils.py
    Args:
        pred_boxes (N, 7): 
        gt_boxes (N, 7): 

    Returns:
        _type_: _description_
    """
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])  # (N, 4, 2)
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])  # (N, 4, 2)   

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # print(volume_union)
    # print(volume_inter)
    print(volume_inter / volume_union)

    return pred_boxes[torch.argmax(volume_inter / volume_union)]


def get_bounding_box(file):
    pc_array = np.load(file)
    #print (pc_array)
    vehicle_points = pc_array[pc_array[:,3] == 6.0]
    vehicle_points = vehicle_points[:, :-1]
    pcd = roi_filter(vehicle_points)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    centers = aabb.get_center()
    extent = aabb.get_extent()
    #print(centers)
    #print(extent)
    #test_box = np.array([[ 9.3091,  5.0264, -1.5887,  4.0584,  1.6776,  1.5724,  6.2854], [ 9.3091,  5.0264, -1.5887,  4.0584,  1.6776,  1.5724,  6.2854]])
    converted_bbox = np.concatenate((centers, extent), axis = 0)
    converted_bbox = np.append(converted_bbox, 0.0)
    converted_bbox = converted_bbox[np.newaxis,:]
    #converted_bbox = np.tile(converted_bbox, (test_box.shape[0],1))
    
    #pred_boxes = torch.tensor(converted_bbox, dtype=torch.float32)
    #gt_boxes = torch.tensor(test_box, dtype=torch.float32)
    print(converted_bbox)
    #bbox3d_overlaps_diou(pred_boxes, gt_boxes)
    # o3d.visualization.draw_geometries([pcd, aabb, obb])
    return converted_bbox

# file = '../../output_data_converted/0-10/HDL-64E/clear/4_labeled_verification.npy'
# get_bounding_box(file)