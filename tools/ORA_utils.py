from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
import torch
import numpy as np
from copy import copy

def cart2sph(x, y, z):
    # Transform Cartesian to spherical coordinates
    hxy = torch.hypot(x, y)
    r = torch.hypot(hxy, z)
    el = torch.arctan2(z, hxy)
    az = torch.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    # Transform spherical to Cartesian coordinates
    rcos_theta = r * torch.cos(el)
    x = rcos_theta * torch.cos(az)
    y = rcos_theta * torch.sin(az)
    z = r * torch.sin(el)
    return x, y, z

def ray_shifting(point_to_be_shifted, shifting_distance):
    # point_to_be_shifted: (x,y,z,intensity)
    # shifting_distance: float, shifting distance in the ray direction.

    origin = torch.tensor([0,0,0])
    # Calculate delt x,y,z
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

    # print(shifted_x)    
    # print(shifted_y)
    # print(shifted_z)
    # print(shifted_intensity)
    shifted_point = torch.tensor([shifted_x, shifted_y, shifted_z, shifted_intensity])
    return shifted_point

def get_point_mask_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)

    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    return points, point_masks, is_numpy

def pick_random_points_in_mask(point_masks, budget):
   # Get the indices of non-zero values
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        # If there are no non-zero indices, return an empty tensor
        return torch.tensor([], dtype=torch.long)
    
    if non_zero_indices.numel() == 1:
        non_zero_indices = torch.tensor([non_zero_indices.item()])
    
    # Shuffle the indices
    shuffled_indices = torch.randperm(non_zero_indices.numel())

    # Select the first k shuffled indices
    k = min(budget, shuffled_indices.numel())  # Choose the number of non-zero values to keep
    selected_indices = shuffled_indices[:k]

    # Create a mask to set non-selected indices to zero
    mask = torch.zeros_like(non_zero_indices, dtype=torch.bool)
    mask[selected_indices] = True

    # # Set non-selected non-zero values to zero
    # point_masks[:, non_zero_indices[mask == False]] = 0

    return non_zero_indices[mask]

def pick_highest_intensity_points_in_mask(points, point_masks, budget):
    
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        # If there are no non-zero indices, return an empty tensor
        return torch.tensor([], dtype=torch.long)

    if non_zero_indices.numel() == 1:
        non_zero_indices = torch.tensor([non_zero_indices.item()])
    # Apply the mask to the full array to get values from the specified column
    column_values = points[non_zero_indices][:, 3]

    # Sort indices based on the values in the specified column
    sorted_indices = non_zero_indices[column_values.argsort()]

    # Select the first k shuffled indices
    k = min(budget, sorted_indices.numel())  # Choose the number of non-zero values to keep
    selected_indices = sorted_indices[:k]

    return selected_indices

def pick_lowest_distance_in_mask(points, point_masks, budget):
    # Get the indices of non-zero values in the mask
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        # If there are no non-zero indices, return an empty tensor
        return torch.tensor([], dtype=torch.long)
    
    if non_zero_indices.numel() == 1:
        non_zero_indices = torch.tensor([non_zero_indices.item()])

    # Extract x, y, z coordinates from points for non-zero indices
    xyz_coordinates = points[non_zero_indices, :3]  # Assuming points has shape [N, 3] for x, y, z

    # Calculate distances from each point to the origin (0, 0, 0)
    distances = torch.norm(xyz_coordinates, dim=1)

    # Sort non-zero indices based on distances
    sorted_indices = non_zero_indices[distances.argsort()]

    # Select the first 'budget' indices (up to 'k' indices if fewer non-zero indices than 'budget')
    k = min(budget, sorted_indices.numel())
    selected_indices = sorted_indices[:k]

    return selected_indices

def get_all_points_in_mask_distance(points, point_masks):
    # Get the indices of non-zero values in the mask
    non_zero_indices = point_masks.squeeze().nonzero().squeeze()

    if non_zero_indices.numel() == 0:
        # If there are no non-zero indices, return an empty tensor
        return torch.tensor([], dtype=torch.long)

    # Extract x, y, z coordinates from points for non-zero indices
    xyz_coordinates = points[non_zero_indices, :3]  # Assuming points has shape [N, 3] for x, y, z

    # Calculate distances from each point to the origin (0, 0, 0)
    distances = torch.norm(xyz_coordinates, dim=1)

    # Sort non-zero indices based on distances
    sorted_indices = non_zero_indices[distances.argsort()]

    return sorted_indices

def shift_selected_points(points, selected_points, shifting_distance):
    # Iterate over each non-zero point in the mask and apply ray_shifting function
    modified_points = copy(points)
    for idx in selected_points:
        # Extract point coordinates and intensity
        point = modified_points[idx]
        x, y, z, intensity = point[0], point[1], point[2], point[3]

        # Apply ray_shifting function to the point
        shifted_point = ray_shifting(torch.tensor([x, y, z, intensity]), shifting_distance)

        # Update the points tensor with the shifted point
        modified_points[idx] = shifted_point
    
    return modified_points

def scale_indices(individual, data_length, max_length):
    scale_factor = data_length / max_length
    scaled_indices = set()
    for idx in individual:
        proposed_index = int(idx * scale_factor)
        while (proposed_index in scaled_indices) and len(scaled_indices) < data_length:
            proposed_index = (proposed_index + 1) % data_length  # Wrap around if necessary
        scaled_indices.add(proposed_index)
    return list(scaled_indices)

def apply_random_ORA_points_in_boxes3d(points, boxes3d, budget, shifting_distance = 2):
    
    points, point_masks, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)

    selected_points = pick_random_points_in_mask(point_masks, budget)

    points = shift_selected_points(points, selected_points, shifting_distance)

    return points.numpy() if is_numpy else points

def apply_intensity_ORA_points_in_boxes3d(points, boxes3d, budget, shifting_distance = 2):
    
    points, point_masks, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)

    selected_points = pick_highest_intensity_points_in_mask(points, point_masks, budget)

    points = shift_selected_points(points, selected_points, shifting_distance)

    return points.numpy() if is_numpy else points

def apply_distance_ORA_points_in_boxes3d(points, boxes3d, budget, shifting_distance = -2):
    
    points, point_masks, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)

    selected_points = pick_lowest_distance_in_mask(points, point_masks, budget)

    points = shift_selected_points(points, selected_points, shifting_distance)

    return points.numpy() if is_numpy else points

def apply_ORA_pre_selected_points(points, selected_points, boxes3d, max_length, shifting_distance = -2):

    points, points_in_bbox, is_numpy = get_point_mask_in_boxes3d(points, boxes3d)
    
    non_zero_indices = points_in_bbox.squeeze().nonzero().squeeze().numpy()

    selected_points = scale_indices(selected_points, len(non_zero_indices), max_length)

    points = shift_selected_points(points, non_zero_indices[selected_points], shifting_distance)

    return points.numpy() if is_numpy else points
