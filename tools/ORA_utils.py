from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
import torch
import numpy as np
from copy import copy
import os

# General ORA utils

def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical coordinates.
    """
    hxy = torch.hypot(x, y)
    r = torch.hypot(hxy, z)
    el = torch.arctan2(z, hxy)
    az = torch.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    """
    Transform spherical coordinates to Cartesian coordinates.
    """
    rcos_theta = r * torch.cos(el)
    x = rcos_theta * torch.cos(az)
    y = rcos_theta * torch.sin(az)
    z = r * torch.sin(el)
    return x, y, z

def ray_shifting(point_to_be_shifted, shifting_distance):
    """
    Shift a point along the ray direction by a specified distance.
    
    Args:
        point_to_be_shifted (tensor): Point (x, y, z, intensity) to be shifted.
        shifting_distance (float): Distance to shift the point along the ray direction.
        
    Returns:
        tensor: Shifted point (x, y, z, intensity).
    """
    origin = torch.tensor([0, 0, 0])
    delt_x = point_to_be_shifted[0] - origin[0]
    delt_y = point_to_be_shifted[1] - origin[1]
    delt_z = point_to_be_shifted[2] - origin[2]

    az, el, r = cart2sph(delt_x, delt_y, delt_z)
    shifted_r = r + shifting_distance
    shifted_delt_x, shifted_delt_y, shifted_delt_z = sph2cart(az, el, shifted_r)

    shifted_x = shifted_delt_x + origin[0]
    shifted_y = shifted_delt_y + origin[1]
    shifted_z = shifted_delt_z + origin[2]
    shifted_intensity = point_to_be_shifted[3]

    shifted_point = torch.tensor([shifted_x, shifted_y, shifted_z, shifted_intensity])
    return shifted_point


def get_point_mask_in_boxes3d(points, boxes3d):
    """
    Get a mask of points that are within given 3D boxes.
    
    Args:
        points (tensor): Points (num_points, 3 + C).
        boxes3d (tensor): 3D boxes (N, 7) [x, y, z, dx, dy, dz, heading].
        
    Returns:
        tuple: Points, point masks, and a flag indicating if input was numpy array.
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)

    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    return points, point_masks, is_numpy

def pick_random_points_in_mask(point_masks, budget):
    """
    Pick random points from a mask.
    
    Args:
        point_masks (tensor): Mask of points.
        budget (int): Number of points to select.
        
    Returns:
        tensor: Indices of selected points.
    """
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    
    if non_zero_indices.numel() == 1:
        non_zero_indices = torch.tensor([non_zero_indices.item()])
    
    shuffled_indices = torch.randperm(non_zero_indices.numel())
    k = min(budget, shuffled_indices.numel())
    selected_indices = shuffled_indices[:k]

    mask = torch.zeros_like(non_zero_indices, dtype=torch.bool)
    mask[selected_indices] = True

    return non_zero_indices[mask]

def pick_highest_intensity_points_in_mask(points, point_masks, budget):
    """
    Pick points with the highest intensity from a mask.
    
    Args:
        points (tensor): Points (num_points, 3 + intensity).
        point_masks (tensor): Mask of points.
        budget (int): Number of points to select.
        
    Returns:
        tensor: Indices of selected points.
    """
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    if non_zero_indices.numel() == 1:
        non_zero_indices = torch.tensor([non_zero_indices.item()])

    column_values = points[non_zero_indices][:, 3]
    sorted_indices = non_zero_indices[column_values.argsort()]
    k = min(budget, sorted_indices.numel())
    selected_indices = sorted_indices[:k]

    return selected_indices

def pick_lowest_distance_in_mask(points, point_masks, budget):
    """
    Pick points with the lowest distance to the origin from a mask.
    
    Args:
        points (tensor): Points (num_points, 3).
        point_masks (tensor): Mask of points.
        budget (int): Number of points to select.
        
    Returns:
        tensor: Indices of selected points.
    """
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    
    if non_zero_indices.numel() == 1:
        non_zero_indices = torch.tensor([non_zero_indices.item()])

    xyz_coordinates = points[non_zero_indices, :3]
    distances = torch.norm(xyz_coordinates, dim=1)
    sorted_indices = non_zero_indices[distances.argsort()]
    k = min(budget, sorted_indices.numel())
    selected_indices = sorted_indices[:k]

    return selected_indices

def get_all_points_in_mask_distance(points, point_masks):
    """
    Get all points in the mask sorted by distance to the origin.
    
    Args:
        points (tensor): Points (num_points, 3).
        point_masks (tensor): Mask of points.
        
    Returns:
        tensor: Indices of points sorted by distance.
    """
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    xyz_coordinates = points[non_zero_indices, :3]
    distances = torch.norm(xyz_coordinates, dim=1)
    sorted_indices = non_zero_indices[distances.argsort()]

    return sorted_indices

def shift_selected_points(points, selected_points, shifting_distance):
    """
    Shift selected points by a specified distance along the ray direction.
    
    Args:
        points (tensor): Points (num_points, 4).
        selected_points (tensor): Indices of points to be shifted.
        shifting_distance (float): Distance to shift the points.
        
    Returns:
        tensor: Points with selected points shifted.
    """
    modified_points = copy(points)
    for idx in selected_points:
        point = modified_points[idx]
        x, y, z, intensity = point[0], point[1], point[2], point[3]
        shifted_point = ray_shifting(torch.tensor([x, y, z, intensity]), shifting_distance)
        modified_points[idx] = shifted_point
    
    return modified_points

def scale_indices(individual, data_length, max_length):
    """
    Scale indices to match the data length and maximum length.
    
    Args:
        individual (list): List of indices.
        data_length (int): Length of the data.
        max_length (int): Maximum length for scaling.
        
    Returns:
        list: Scaled indices.
    """
    scale_factor = data_length / max_length
    scaled_indices = set()
    for idx in individual:
        proposed_index = int(idx * scale_factor)
        while (proposed_index in scaled_indices) and len(scaled_indices) < data_length:
            proposed_index = (proposed_index + 1) % data_length
        scaled_indices.add(proposed_index)
    return list(scaled_indices)

def apply_random_ORA_points_in_boxes3d(points, boxes3d, budget, shifting_distance=2):
    """
    Apply random ORA (Object Removal Attacks) to points within 3D boxes.
    
    Args:
        points (tensor): Points (num_points, 4).
        boxes3d (tensor): 3D boxes (N, 7).
        budget (int): Number of points to shift.
        shifting_distance (float): Distance to shift the points.
        
    Returns:
        tensor: Points with ORA applied.
    """
    points, point_masks, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)
    selected_points = pick_random_points_in_mask(point_masks, budget)
    points = shift_selected_points(points, selected_points, shifting_distance)
    return points.numpy() if is_numpy else points

def apply_intensity_ORA_points_in_boxes3d(points, boxes3d, budget, shifting_distance=2):
    """
    Apply intensity-based ORA (Object Removal Attacks) to points within 3D boxes.
    
    Args:
        points (tensor): Points (num_points, 4).
        boxes3d (tensor): 3D boxes (N, 7).
        budget (int): Number of points to shift.
        shifting_distance (float): Distance to shift the points.
        
    Returns:
        tensor: Points with ORA applied.
    """
    points, point_masks, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)
    selected_points = pick_highest_intensity_points_in_mask(points, point_masks, budget)
    points = shift_selected_points(points, selected_points, shifting_distance)
    return points.numpy() if is_numpy else points

def apply_distance_ORA_points_in_boxes3d(points, boxes3d, budget, shifting_distance=2):
    """
    Apply distance-based ORA (Object Removal Attacks) to points within 3D boxes.
    
    Args:
        points (tensor): Points (num_points, 4).
        boxes3d (tensor): 3D boxes (N, 7).
        budget (int): Number of points to shift.
        shifting_distance (float): Distance to shift the points.
        
    Returns:
        tensor: Points with ORA applied.
    """
    points, point_masks, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)
    selected_points = pick_lowest_distance_in_mask(points, point_masks, budget)
    points = shift_selected_points(points, selected_points, shifting_distance)
    return points.numpy() if is_numpy else points

def apply_ORA_pre_selected_points(points, selected_points, boxes3d, max_length, shifting_distance=-2):
    """
    Apply ORA (Object Removal Attacks) to pre-selected points within 3D boxes.
    
    Args:
        points (tensor): Points (num_points, 4).
        selected_points (list): List of selected points.
        boxes3d (tensor): 3D boxes (N, 7).
        max_length (int): Maximum length for scaling.
        shifting_distance (float): Distance to shift the points.
        
    Returns:
        tensor: Points with ORA applied.
    """
    points, points_in_bbox, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)
    non_zero_indices = points_in_bbox.squeeze().nonzero().squeeze().numpy()
    selected_points = scale_indices(selected_points, len(non_zero_indices), max_length)
    points = shift_selected_points(points, non_zero_indices[selected_points], shifting_distance)
    return points.numpy() if is_numpy else points
