import cv2
import numpy as np
import torch
import os
from pathlib import Path


def create_mask(frame_gray):
    # Create a mask to avoid detection in specularities in the image (bright spots)
    kernel = np.ones((5, 5), np.uint8)
    thresh_bright = cv2.threshold(frame_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    thresh_bright = cv2.erode(thresh_bright, kernel, iterations=5)
    # Create a mask to avoid detection in dark spots
    thresh_dark = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh_dark = cv2.erode(thresh_dark, kernel, iterations=3)
    # Combine both masks
    thresh = cv2.bitwise_and(thresh_bright, thresh_dark)
    return thresh

#TODO: Add bright thresh and dark thresh as parameters and modify the pipeline accordingly
def create_mask_normalized(frame_gray_norm):
    """
    Create a mask for specularities for images normalized to [0,1].
    The bright threshold 220/255 (~0.86) is inverted and the dark threshold 10/255 (~0.04) is applied.
    """
    kernel = np.ones((5, 5), np.float32)
    # Threshold for bright spots (inverted)
    thresh_bright = cv2.threshold(frame_gray_norm, 220/255.0, 1, cv2.THRESH_BINARY_INV)[1]
    thresh_bright = cv2.erode(thresh_bright, kernel, iterations=5)
    # Threshold for dark spots
    thresh_dark = cv2.threshold(frame_gray_norm, 10/255.0, 1, cv2.THRESH_BINARY)[1]
    thresh_dark = cv2.erode(thresh_dark, kernel, iterations=3)
    # Combine both masks
    thresh = cv2.bitwise_and(thresh_bright, thresh_dark)
    # Convert mask values from {0,1} to {0,255} as uint8
    return thresh.astype(np.uint8)

def is_in_mask(points: torch.Tensor, mask_points: torch.Tensor) -> torch.Tensor:
    """
    Given points: (K,2) and mask_points: (N,2) (both in (col, row) order),
    returns a boolean tensor of shape (K,) where each element is True if the corresponding
    point is found exactly in mask_points.
    """
    
    # Remove extra dimensions if present (e.g. if shape is (1, K, 2))
    if points.ndim > 2:
        points = points.squeeze(0)
    if mask_points.ndim > 2:
        mask_points = mask_points.squeeze(0)

    # Check if there are no points in input
    if points.size(0) == 0:
        return torch.empty((0,), dtype=torch.bool, device=points.device)
    # Compare each point (K,2) with each mask point (N,2).
    equal = torch.all(points[:, None, :] == mask_points[None, :, :], dim=2)  # shape: (K,N)
    return torch.any(equal, dim=1)


def get_bgr_image(img):
    """Convert a PyTorch tensor image (C,H,W) to a NumPy BGR image."""
    return cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

def numpy_mask_to_tensor(mask_np: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy mask (2D array, dtype=np.uint8) to a torch tensor
    with shape (1, H, W) and dtype=torch.uint8.
    """
    if mask_np.ndim == 2:
        mask_np = np.expand_dims(mask_np, axis=0)  # shape: (1, H, W)
    tensor_mask = torch.from_numpy(mask_np)
    return tensor_mask 

def get_mask_and_masked_image(img_np):
    """Return the mask and the masked image (in RGB) given a NumPy BGR image."""
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    mask = create_mask_normalized(img_gray)
    mask = mask.astype(np.uint8)
    # Apply the mask to get the masked image in BGR
    masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)
    # Convert the masked image from BGR to RGB
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    return mask, masked_img

def get_mask_points(mask: torch.Tensor) -> torch.Tensor:
    """
    Given a mask tensor of shape (1, H, W) (dtype=torch.uint8), where the mask values
    of interest are 0, return a tensor of shape (N,2) with points in (col, row) order.
    """
    # Make sure mask has a single channel; if not, squeeze extra dimensions.
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)  # now shape (H, W)
    # Get indices where mask == 0. nonzero() returns (row, col) candidates.
    pts = torch.nonzero(mask == 0, as_tuple=False)  # shape (N,2); order: (row, col)
    # Convert (row, col) to (col, row) = (x, y) order
    mask_points = pts[:, [1, 0]].unsqueeze(0)  # shape (1, N, 2)
    return mask_points

def filter_feats_by_mask(kpts: torch.Tensor, desc: torch.Tensor, mask_points: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Given keypoints (1, N, 2), descriptors (1, N, D) and mask_points (1, M, 2) in (col, row) order,
    returns the filtered keypoints and descriptors using the same valid index mask.
    """
    # Remove the batch dimension for processing
    kpts_unbatched = kpts.squeeze(0)       # shape: (N, 2)
    desc_unbatched = desc.squeeze(0)       # shape: (N, D)
    mask_points = mask_points.squeeze(0)   # shape: (M, 2)

    # Compute valid mask from keypoints (using floor and ceil)
    kpts_floor = torch.floor(kpts_unbatched).long()
    kpts_ceil  = torch.ceil(kpts_unbatched).long()
    valid = ~(is_in_mask(kpts_floor, mask_points) | is_in_mask(kpts_ceil, mask_points))
    
    # Filter both keypoints and descriptors using the same valid mask
    kpts_filtered = kpts_unbatched[valid].unsqueeze(0)     # shape: (1, N_filtered, 2)
    desc_filtered = desc_unbatched[valid].unsqueeze(0)     # shape: (1, N_filtered, D)
    
    return kpts_filtered, desc_filtered


# TODO: Old, for numpy arrays. Remove if not needed.
def filter_keypoints_pair(kpts0, kpts1, mask0_points, mask1_points):
    """
    Filter a pair of keypoints arrays using the corresponding mask points.
    Returns filtered keypoints0 and keypoints1.
    """
    valid0 = filter_keypoints_by_mask(kpts0, mask0_points)
    valid1 = filter_keypoints_by_mask(kpts1, mask1_points)
    valid = valid0 & valid1
    return kpts0[valid], kpts1[valid]

# TODO: Old, for numpy arrays. Remove if not needed.
def filter_result(result, mask0_points, mask1_points):
    """
    Returns a new result dict with matched and inlier keypoints filtered 
    using the provided mask points.
    """
    mkpts0, mkpts1 = filter_keypoints_pair(result['matched_kpts0'],
                                             result['matched_kpts1'],
                                             mask0_points, mask1_points)
    ikpts0, ikpts1 = filter_keypoints_pair(result['inlier_kpts0'],
                                             result['inlier_kpts1'],
                                             mask0_points, mask1_points)
    result_filtered = result.copy()
    result_filtered['matched_kpts0'] = mkpts0
    result_filtered['matched_kpts1'] = mkpts1
    result_filtered['inlier_kpts0'] = ikpts0
    result_filtered['inlier_kpts1'] = ikpts1
    return result_filtered

#TODO: Old, for numpy arrays. Remove if not needed.
def get_mask_points_and_masked(img_np):
    """
    Given a BGR image (as NumPy array), compute its grayscale image, mask (using create_mask_normalized),
    the masked image and return the mask zero points in (col, row) order along with the masked image.
    """
    # Convert the image to grayscale to create the mask
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # Create the mask and apply it to the image
    mask = create_mask_normalized(img_gray)
    # Apply the mask to the image
    masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)
    # Clip the masked image to [0,1] range
    masked_img = np.clip(masked_img, 0, 1)
    # Convert the masked image from BGR to RGB for plotting
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    # Get the mask zero points in (col, row) order
    mask_zero = np.where(mask == 0)
    mask_zero_points = np.column_stack((mask_zero[1], mask_zero[0]))
    return mask_zero_points, masked_img
