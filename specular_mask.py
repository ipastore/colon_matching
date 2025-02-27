import cv2
import numpy as np
import torch
import os
from pathlib import Path
from matching.utils import get_default_device, to_tensor, to_numpy
import logging

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

def is_in_mask(points: torch.Tensor, mask_points: torch.Tensor, logger: logging.Logger=None) -> torch.Tensor:
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

#TODO: old, modified with an inline torch function within process_pairs.py
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
    mask_points = pts[:, [1, 0]].unsqueeze(0)  # shape: (1, N, 2)
    return mask_points

def filter_feats_by_mask(kpts: torch.Tensor, desc: torch.Tensor, mask_points: torch.Tensor, logger: logging.Logger=None) -> (torch.Tensor, torch.Tensor):
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
    valid = ~(is_in_mask(kpts_floor, mask_points, logger) | is_in_mask(kpts_ceil, mask_points, logger))
    
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

# #TODO: tidy up
# def filter_image_feats_with_mask(img: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray,
#                                  kpt: torch.Tensor | np.ndarray | tuple, desc: torch.Tensor | np.ndarray| tuple,
#                                  logger=None) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
#     """
#     This function accepts inputs as either torch.Tensor, np.ndarray, or tuple (for keypoints as cv2.KeyPoint)
#     and returns outputs in the same type. It filters the feature keypoints and descriptors using a mask,
#     ensuring the sizes of the image and mask are compatible.
#     """
#     feat_numpy_tuple= False
#     device = get_default_device()

#     if logger: 
#         logger.debug("Starting filter_image_feats_with_mask")
#         logger.debug(f'img type: {type(img)}; img shape: {img.shape}; dtype: {img.dtype}')
#         logger.debug(f'mask type: {type(mask)}; mask shape: {mask.shape}; dtype: {mask.dtype}')
#         logger.debug(f'kpt type: {type(kpt)}; kpt shape: {kpt.shape if isinstance(kpt, torch.Tensor) else len(kpt)}')
#         logger.debug(f'desc type: {type(desc)}; desc shape: {desc.shape if isinstance(desc, torch.Tensor) else len(desc)}')
    
#     # Ensure mask is a torch.Tensor with batch dimension.
#     if isinstance(mask, np.ndarray):
#         mask = torch.from_numpy(mask).unsqueeze(0)
#     # Convert image if provided as np.ndarray.
#     if isinstance(img, np.ndarray):
#         img = torch.from_numpy(img).unsqueeze(0)

#     # Handle keypoints provided as a tuple of cv2.KeyPoint.
#     if isinstance(kpt, (tuple, list)) and len(kpt) > 0 and hasattr(kpt[0], 'pt'):
#         new_kpt = np.array([kp.pt for kp in kpt], dtype=np.float32)
#         # new_kpt = new_kpt[:, [1, 0]]  # Convert (x,y) to (col,row) order
#         kpt = torch.from_numpy(new_kpt).unsqueeze(0).to(device)
#         feat_numpy_tuple = True
#         if isinstance(desc, np.ndarray):
#             desc = torch.from_numpy(desc).unsqueeze(0).to(device)
#         elif isinstance(desc, (tuple, list)):
#             desc = torch.tensor(desc).unsqueeze(0).to(device)
#     elif isinstance(kpt, np.ndarray) and isinstance(desc, np.ndarray):
#         kpt = torch.from_numpy(kpt).unsqueeze(0).to(device)
#         desc = torch.from_numpy(desc).unsqueeze(0).to(device)
#         feat_numpy_tuple = True
#     else:
#         kpt = kpt.to(device)
#         desc = desc.to(device)
    
#     if logger: 
#         #Print the max and the min of the keypoints for H in CHW
#         logger.debug(f'kpt max[:,:,0]: {torch.max(kpt[:,:,0])}; kpt min: {torch.min(kpt[:,:,0])}')
#         #Print the max and the min of the keypoints for W in CHW
#         logger.debug(f'kpt[:,:,1] max: {torch.max(kpt[:,:,1])}; kpt min: {torch.min(kpt[:,:,1])}')
        
#     # Apply mask filtering
#     assert img.shape[-2:] == mask.shape[-2:]
#     mask_points = get_mask_points(mask)
#     if logger:
#         #Print the max and the min of the mask points for H in CHW
#         logger.debug(f'Reordering mask points to get (col, row) order')
#         logger.debug(f'mask_points[:,:,0] max: {torch.max(mask_points[:,:,0])}; mask_points min: {torch.min(mask_points[:,:,0])}')
#         #Print the max and the min of the mask points for W in CHW
#         logger.debug(f'mask_points max[:,:,1]: {torch.max(mask_points[:,:,1])}; mask_points min: {torch.min(mask_points[:,:,1])}')
#         logger.debug(f'mask_points shape: {mask_points.shape}')
    
#     # ChecK: kpt and desc should be in col row order (x,y) (W, H)
#     kpt_filtered, desc_filtered = filter_feats_by_mask(kpt, desc, mask_points, logger)
    
#     if logger:
#         logger.debug(f'kpt_filtered shape: {kpt_filtered.shape}; desc_filtered shape: {desc_filtered.shape}')
#         logger.debug(f'Filtered kpts: {kpt.shape[1]} -> {kpt_filtered.shape[1]}')
#         logger.debug(f'kpt_filtered.device: {kpt_filtered.device}; desc_filtered.device: {desc_filtered.device}')

#     # Convert back to numpy if the input was a tuple of cv2.KeyPoint
#     if feat_numpy_tuple:
#         kpt_filtered = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpt_filtered.squeeze(0).cpu().numpy()]
#         desc_filtered = desc_filtered.squeeze(0).cpu().numpy()
#         if logger:
#             logger.debug(f'kpt_filtered.device: cpu ; desc_filtered.device: cpu')
#             logger.debug(f'Converted to numpy: kpt_filtered shape: {len(kpt_filtered)}; desc_filtered shape: {desc_filtered.shape}')

#     return kpt_filtered, desc_filtered

# A helper for converting cv2.KeyPoint list/tuple to tensor.
def keypoints_to_tensor(kpts, device):
    pts = np.array([kp.pt for kp in kpts], dtype=np.float32)
    return to_tensor(pts, device=device).unsqueeze(0)

def tensor_to_keypoints(kpt_tensor):
    kpt_np = to_numpy(kpt_tensor.squeeze(0))
    return [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpt_np]

def get_dtype_of_collection(collection):
    if isinstance(collection, np.ndarray) or isinstance(collection, torch.Tensor):
        return collection.dtype
    elif isinstance(collection, (list, tuple)) and len(collection) > 0:
        first_element = collection[0]
        if hasattr(first_element, 'dtype'):
            return first_element.dtype
        else:
            return type(first_element)
    

def filter_image_feats_with_mask(img, mask, kpt, desc, logger=None):
    """
    Filter feature keypoints and descriptors using a mask.
    Accepts img, mask, kpt, and desc as torch.Tensor, np.ndarray, or
    for keypoints, a list/tuple of cv2.KeyPoint.
    Returns outputs in the same type as keypoints input.
    """
    device = get_default_device()
    as_cv2_keypoints = False

    if logger:
        logger.debug("Starting filter_image_feats_with_mask")
        logger.debug(f'img type: {type(img)}; img shape: {img.shape}; dtype: {img.dtype}')
        logger.debug(f'mask type: {type(mask)}; mask shape: {mask.shape}; dtype: {mask.dtype}')
        logger.debug(f'kpt type: {type(kpt)}; kpt shape: {kpt.shape if isinstance(kpt, torch.Tensor) else len(kpt)}; dtype: {get_dtype_of_collection(kpt)}')
        logger.debug(f'desc type: {type(desc)}; desc shape: {desc.shape if isinstance(desc, torch.Tensor) else len(desc)}; dtype: {get_dtype_of_collection(desc)}')

    # Convert img and mask to torch.Tensor, adding a batch dimension if needed.
    if isinstance(img, np.ndarray):
        img_tensor = to_tensor(img, device=device).unsqueeze(0)
    else:
        img_tensor = img.to(device)
        
    if isinstance(mask, np.ndarray):
        mask_tensor = to_tensor(mask, device=device).unsqueeze(0)
    else:
        mask_tensor = mask.to(device)
    
    # Process keypoints and descriptors.
    # If keypoints are provided as a list/tuple of cv2.KeyPoint.
    if isinstance(kpt, (list, tuple)) and len(kpt) > 0 and hasattr(kpt[0], 'pt'):
        as_cv2_keypoints = True
        kpt_tensor = keypoints_to_tensor(kpt, device)
        if isinstance(desc, np.ndarray):
            desc_tensor = to_tensor(desc, device=device).unsqueeze(0)
        elif isinstance(desc, (list, tuple)):
            desc_tensor = to_tensor(np.array(desc), device=device).unsqueeze(0)
    elif isinstance(kpt, np.ndarray) and isinstance(desc, np.ndarray):
        kpt_tensor = to_tensor(kpt, device=device).unsqueeze(0)
        desc_tensor = to_tensor(desc, device=device).unsqueeze(0)
        as_cv2_keypoints = True
    else:
        # Assume they are already torch.Tensors.
        kpt_tensor = kpt.to(device)
        desc_tensor = desc.to(device)

    if logger:
        logger.debug(f"Converted keypoints tensor shape: {kpt_tensor.shape}")
        logger.debug(f"Converted descriptors tensor shape: {desc_tensor.shape}")

    # Ensure the spatial dimensions of img and mask match.
    assert img_tensor.shape[-2:] == mask_tensor.shape[-2:], "Image and mask spatial dimensions must match"

    # Obtain mask points: Inside the coordinates of the mask_points to (x,y) (col,row) (W, H) order.
    mask_points = get_mask_points(mask_tensor)
    if logger:
        logger.debug(f"mask_points shape: {mask_points.shape}")
    
    #TODO: Check if any max of kpts are greater thant the img coordinates. WARNING to DEBUG and change coordinates!
        kpt_0_max = torch.max(kpt_tensor[:,:,0])
        kpt_1_max = torch.max(kpt_tensor[:,:,1])
        img_0_max = img_tensor.shape[-1]
        img_1_max = img_tensor.shape[-2]
        mask0_max = torch.max(mask_points[:,:,0])
        mask1_max = torch.max(mask_points[:,:,1])

        if logger: 
            if kpt_0_max > img_0_max or kpt_1_max > img_1_max or mask0_max > img_0_max or mask1_max > img_1_max:
                logger.warning(f'img_0_max: {img_0_max}; img_1_max: {img_1_max}')
                logger.warning(f'kpt_0_max: {kpt_0_max}; kpt_1_max: {kpt_1_max}')
                logger.warning(f'mask0_max: {mask0_max}; mask1_max: {mask1_max}')
                logger.warning(f'Check if the keypoints or maskpoints are in the right order (x,y) or (col,row)')
            else:
                logger.debug(f'img_0_max: {img_0_max}; img_1_max: {img_1_max}')
                logger.debug(f'kpt_0_max: {kpt_0_max}; kpt_1_max: {kpt_1_max}')
                logger.debug(f'mask0_max: {mask0_max}; mask1_max: {mask1_max}')
                 
    # Filter features using your existing filtering function.
    kpt_filtered, desc_filtered = filter_feats_by_mask(kpt_tensor, desc_tensor, mask_points, logger)
    
    if logger:
        logger.debug(f"Filtered keypoints shape: {kpt_filtered.shape}")
        logger.debug(f"Filtered descriptors shape: {desc_filtered.shape}")

    # Convert back to original format if keypoints were initially cv2.KeyPoint objects.
    # Inside both functions, the kpts and descs are sent to cpu
    if as_cv2_keypoints:
        kpt_filtered = tensor_to_keypoints(kpt_filtered)
        desc_filtered = to_numpy(desc_filtered.squeeze(0))
    
    return kpt_filtered, desc_filtered