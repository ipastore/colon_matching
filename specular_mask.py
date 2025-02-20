import cv2
import numpy as np
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

def create_mask_normalized(frame_gray_norm):
    """
    Create a mask for specularities for images normalized to [0,1].
    The bright threshold 220/255 (~0.86) is inverted and the dark threshold 10/255 (~0.04) is applied.
    """
    kernel = np.ones((5, 5), np.float32)
    # Threshold for bright spots (inverted)
    thresh_bright = cv2.threshold(frame_gray_norm, 210/255.0, 1, cv2.THRESH_BINARY_INV)[1]
    thresh_bright = cv2.erode(thresh_bright, kernel, iterations=5)
    # Threshold for dark spots
    thresh_dark = cv2.threshold(frame_gray_norm, 10/255.0, 1, cv2.THRESH_BINARY)[1]
    thresh_dark = cv2.erode(thresh_dark, kernel, iterations=3)
    # Combine both masks
    thresh = cv2.bitwise_and(thresh_bright, thresh_dark)
    # Convert mask values from {0,1} to {0,255} as uint8
    return (thresh * 255).astype(np.uint8)

def is_in_mask(points, mask_points):
    """
    points: (K,2), mask_points: (N,2)
    Returns a boolean array of shape (K,) where True if a point is in mask_points.
    """
    if points.size == 0:
        return np.empty((0,), dtype=bool)
    return np.any(np.all(points[:, None] == mask_points[None, :], axis=2), axis=1)

def get_bgr_image(img):
    """Convert a PyTorch tensor image (C,H,W) to a NumPy BGR image."""
    return cv2.cvtColor(img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

def get_mask_and_masked_image(img_np):
    """Return the mask and the masked image given a NumPy BGR image."""
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    mask = create_mask_normalized(img_gray)
    masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)
    return mask, masked_img

def filter_keypoints_by_mask(kpts, mask_points):
    """
    Given keypoints (N,2) and mask_points (M,2) in (col, row) order,
    return a boolean array of valid keypoints not in the mask.
    """
    kpts_floor = np.floor(kpts).astype(int)
    kpts_ceil  = np.ceil(kpts).astype(int)
    valid = ~(is_in_mask(kpts_floor, mask_points) | is_in_mask(kpts_ceil, mask_points))
    return valid

def filter_keypoints_pair(kpts0, kpts1, mask0_points, mask1_points):
    """
    Filter a pair of keypoints arrays using the corresponding mask points.
    Returns filtered keypoints0 and keypoints1.
    """
    valid0 = filter_keypoints_by_mask(kpts0, mask0_points)
    valid1 = filter_keypoints_by_mask(kpts1, mask1_points)
    valid = valid0 & valid1
    return kpts0[valid], kpts1[valid]

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
