import cv2
import numpy as np
import torch
import os
from pathlib import Path
from my_logging import debug_log
import gc
from memory_profiler import profile
import sys

def get_default_device():
    device = "cpu"

    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"

    elif torch.cuda.is_available():
        device = "cuda"

    return device


def to_numpy(x: torch.Tensor | np.ndarray | dict | list) -> np.ndarray:
    """convert item or container of items to numpy

    Args:
        x (torch.Tensor | np.ndarray | dict | list): input

    Returns:
        np.ndarray: numpy array of input
    """
    if isinstance(x, list):
        return np.array([to_numpy(i) for i in x])
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = to_numpy(v)
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x


def to_tensor(x: np.ndarray | torch.Tensor, device: str = None) -> torch.Tensor:
    """Convert to tensor and place on device

    Args:
        x (np.ndarray | torch.Tensor): item to convert to tensor
        device (str, optional): device to place tensor on. Defaults to None.

    Returns:
        torch.Tensor: tensor with data from `x` on device `device`
    """
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if device is not None:
        return x.to(device)


def to_normalized_coords(pts: np.ndarray | torch.Tensor, height: int, width: int):
    """normalize kpt coords from px space to [0,1]
    Assumes pts are in x, y order in array/tensor shape (N, 2)

    Args:
        pts (np.ndarray | torch.Tensor): array of kpts, must be shape (N, 2)
        height (int): height of img
        width (int): width of img

    Returns:
        np.array: kpts in normalized [0,1] coords
    """
    # normalize kpt coords from px space to [0,1]
    # assume pts are in x,y order
    assert pts.shape[-1] == 2, f"input to `to_normalized_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = to_numpy(pts).astype(float)
    pts[:, 0] /= width
    pts[:, 1] /= height

    return pts


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

#TODO colon: Add bright thresh and dark thresh as parameters and modify the pipeline accordingly
def create_mask_normalized(frame_gray_norm):
    """
    Create a mask for specularities for images normalized to [0,1].
    The bright threshold 220/255 (~0.86) is inverted and the dark threshold 10/255 (~0.04) is applied.
    0 = Specularities and dark spots, 1 = everything else.

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

#TODO colopn: Old function, replaced by is_in_mask() but memory efficient with batching  
# def is_in_mask(points: torch.Tensor, mask_points: torch.Tensor, logger=None) -> torch.Tensor:
#     """
#     Given points: (K,2) and mask_points: (N,2) (both in (col, row) order),
#     returns a boolean tensor of shape (K,) where each element is True if the corresponding
#     point is found exactly in mask_points.
#     """

#     # Remove extra dimensions if present (e.g. if shape is (1, K, 2))
#     if points.ndim > 2:
#         points = points.squeeze(0)
#     if mask_points.ndim > 2:
#         mask_points = mask_points.squeeze(0)

#     # Check if there are no points in input
#     if points.size(0) == 0:
#         return torch.empty((0,), dtype=torch.bool, device=points.device)
#     # Compare each point (K,2) with each mask point (N,2).
#     equal = torch.all(points[:, None, :] == mask_points[None, :, :], dim=2)  # shape: (K,N)
#     return torch.any(equal, dim=1)

def is_in_mask(points: torch.Tensor, mask_points: torch.Tensor, logger=None, batch_size=1000, mask_batch_size=10000) -> torch.Tensor:
    """
    Given points: (K,2) and mask_points: (N,2) (both in (col, row) order),
    returns a boolean tensor of shape (K,) where each element is True if the corresponding
    point is found exactly in mask_points.
    
    This implementation processes both points and mask points in batches to reduce memory usage.
    
    Args:
        points: Tensor of shape (K,2) containing points to check
        mask_points: Tensor of shape (N,2) containing mask points
        logger: Optional logger for debugging
        batch_size: Number of points to process at once
        mask_batch_size: Number of mask points to process at once
        
    Returns:
        Boolean tensor of shape (K,) where True indicates point is in mask
    """
    # Remove extra dimensions if present (e.g. if shape is (1, K, 2))
    if points.ndim > 2:
        points = points.squeeze(0)
    if mask_points.ndim > 2:
        mask_points = mask_points.squeeze(0)

    # Check if there are no points in input
    if points.size(0) == 0:
        return torch.empty((0,), dtype=torch.bool, device=points.device)
    
    # Initialize result tensor
    result = torch.zeros(points.size(0), dtype=torch.bool, device=points.device)
    
    # Process points in batches
    for start_idx in range(0, points.size(0), batch_size):
        end_idx = min(start_idx + batch_size, points.size(0))
        # Get current batch of points
        batch_points = points[start_idx:end_idx]
        
        # Process mask points in sub-batches
        batch_result = torch.zeros(batch_points.size(0), dtype=torch.bool, device=points.device)
        
        for mask_start_idx in range(0, mask_points.size(0), mask_batch_size):
            mask_end_idx = min(mask_start_idx + mask_batch_size, mask_points.size(0))
            # Get current batch of mask points
            batch_mask_points = mask_points[mask_start_idx:mask_end_idx]
            
            # Compare batch points with batch mask points
            # Shape: (batch_size, mask_batch_size, 2)
            equality = batch_points[:, None, :] == batch_mask_points[None, :, :]
            
            # Check if all dimensions match (points are identical)
            # Shape: (batch_size, mask_batch_size)
            all_equal = torch.all(equality, dim=2)
            
            # Check if any mask point in this batch matches
            # Shape: (batch_size,)
            matches = torch.any(all_equal, dim=1)
            
            # Update batch results (logical OR with previous results)
            batch_result = batch_result | matches
            
            # Free memory explicitly
            del equality, all_equal, matches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Add batch results to overall result
        result[start_idx:end_idx] = batch_result
        
        # Free memory explicitly
        del batch_points, batch_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result

def get_bgr_image(img):
    """Convert a PyTorch tensor image (C,H,W) to a NumPy BGR image while freeing memory."""
    img_np = cv2.cvtColor(img.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
    
    del img
    return img_np



def get_mask_and_masked_image(img_np):
    """
    Return the mask and the masked image (in RGB) given a NumPy BGR image.
    0 = Specularities and dark spots, 1 = everything else.
    """
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    mask = create_mask_normalized(img_gray).astype(np.uint8)

    # Apply mask **in-place** without creating additional copies
    img_np[:, :, 0] *= mask
    img_np[:, :, 1] *= mask
    img_np[:, :, 2] *= mask

    # Convert to RGB **in-place** to avoid unnecessary duplication
    masked_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Explicitly delete large variables
    del img_gray, img_np

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

def filter_feats_by_mask(kpts: torch.Tensor, desc: torch.Tensor, mask_points: torch.Tensor, logger = None) -> (torch.Tensor, torch.Tensor):
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
    
#TODO colon: Old, to be replaced by filter_feat_dict_with_mask. But still implemented in gim-lg, and sift-nn
def filter_image_feats_with_mask(img, mask, kpt, desc, logger=None):
    """
    Filter feature keypoints and descriptors using a mask.
    Accepts img, mask, kpt, and desc as torch.Tensor, np.ndarray, or
    for keypoints, a list/tuple of cv2.KeyPoint.
    Returns outputs in the same type as keypoints input.
    """
    device = get_default_device()
    as_cv2_keypoints = False

    debug_log(logger, "filter_image_feats_with_mask", "Starting filter_image_feats_with_mask")
    debug_log(logger, "filter_image_feats_with_mask", f'img type: {type(img)}; img shape: {img.shape}; dtype: {img.dtype}')
    debug_log(logger, "filter_image_feats_with_mask", f'mask type: {type(mask)}; mask shape: {mask.shape}; dtype: {mask.dtype}')
    debug_log(logger, "filter_image_feats_with_mask", f'kpt type: {type(kpt)}; kpt shape: {kpt.shape if isinstance(kpt, torch.Tensor) else len(kpt)}; dtype: {get_dtype_of_collection(kpt)}')
    debug_log(logger, "filter_image_feats_with_mask", f'desc type: {type(desc)}; desc shape: {desc.shape if isinstance(desc, torch.Tensor) else len(desc)}; dtype: {get_dtype_of_collection(desc)}')
    
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

    debug_log(logger, "filter_image_feats_with_mask", f"Converted keypoints tensor shape: {kpt_tensor.shape}")
    debug_log(logger, "filter_image_feats_with_mask", f"Converted descriptors tensor shape: {desc_tensor.shape}")
    
    # Ensure the spatial dimensions of img and mask match.
    assert img_tensor.shape[-2:] == mask_tensor.shape[-2:], "Image and mask spatial dimensions must match"

    # Obtain mask points: Inside the coordinates of the mask_points to (x,y) (col,row) (W, H) order.
    mask_points = get_mask_points(mask_tensor)
    debug_log(logger, "filter_image_feats_with_mask", f"mask_points shape: {mask_points.shape}")

    # Early check of the debug flag using the active flags stored on the logger.
    active_flags = getattr(logger, 'active_debug_flags', set())
    if "filter_image_feats_with_mask" in active_flags or "ALL" in active_flags:
        
        kpt_0_max = torch.max(kpt_tensor[:,:,0])
        kpt_1_max = torch.max(kpt_tensor[:,:,1])
        img_0_max = img_tensor.shape[-1]
        img_1_max = img_tensor.shape[-2]
        mask0_max = torch.max(mask_points[:,:,0])
        mask1_max = torch.max(mask_points[:,:,1])

        if kpt_0_max > img_0_max or kpt_1_max > img_1_max or mask0_max > img_0_max or mask1_max > img_1_max:
            logger.warning(f'img_0_max: {img_0_max}; img_1_max: {img_1_max}')
            logger.warning(f'kpt_0_max: {kpt_0_max}; kpt_1_max: {kpt_1_max}')
            logger.warning(f'mask0_max: {mask0_max}; mask1_max: {mask1_max}')
            logger.warning(f'Check if the keypoints or maskpoints are in the right order (x,y) or (col,row)')
        else:
            debug_log(logger, "filter_image_feats_with_mask", f'img_0_max: {img_0_max}; img_1_max: {img_1_max}')
            debug_log(logger, "filter_image_feats_with_mask", f'kpt_0_max: {kpt_0_max}; kpt_1_max: {kpt_1_max}')
            debug_log(logger, "filter_image_feats_with_mask", f'mask0_max: {mask0_max}; mask1_max: {mask1_max}')
            
    # Filter features using your existing filtering function.
    kpt_filtered, desc_filtered = filter_feats_by_mask(kpt_tensor, desc_tensor, mask_points, logger)
    
    debug_log(logger, "filter_image_feats_with_mask", f"Filtered keypoints shape: {kpt_filtered.shape}")
    debug_log(logger, "filter_image_feats_with_mask", f"Filtered descriptors shape: {desc_filtered.shape}")
    
    # Convert back to original format if keypoints were initially cv2.KeyPoint objects.
    # Inside both functions, the kpts and descs are sent to cpu
    if as_cv2_keypoints:
        kpt_filtered = tensor_to_keypoints(kpt_filtered)
        desc_filtered = to_numpy(desc_filtered.squeeze(0))
    
    return kpt_filtered, desc_filtered

def filter_feat_dict_with_mask(img, mask, feats, logger=None):
    """
    Filter an entire features dictionary using a mask.
    
    Args:
        img: Image as torch.Tensor or np.ndarray
        mask: Mask as torch.Tensor or np.ndarray
        feats: Dictionary containing feature data with required keys 'keypoints' and 'descriptors'
        logger: Optional logger for debugging
        
    Returns:
        Dictionary with all features filtered according to the mask
    """
    device = get_default_device()
    as_cv2_keypoints = False
    
    debug_log(logger, "filter_feat_dict_with_mask", "Starting filter_feat_dict_with_mask")
    debug_log(logger, "filter_feat_dict_with_mask", f'img type: {type(img)}; img shape: {img.shape}')
    debug_log(logger, "filter_feat_dict_with_mask", f'mask type: {type(mask)}; mask shape: {mask.shape}')
    debug_log(logger, "filter_feat_dict_with_mask", f'feats keys: {feats.keys()}')
    
    # Check required keys
    required_keys = ['keypoints', 'descriptors']
    for key in required_keys:
        if key not in feats:
            msg = f"Feature dict must contain '{key}' key"
            if logger:
                logger.error(msg)
            raise ValueError(msg)
    
    # Convert img and mask to torch tensors
    if isinstance(img, np.ndarray):
        img_tensor = to_tensor(img, device=device).unsqueeze(0)
    else:
        img_tensor = img.to(device)
        
    if isinstance(mask, np.ndarray):
        mask_tensor = to_tensor(mask, device=device).unsqueeze(0)
    else:
        mask_tensor = mask.to(device)
    
    # Process keypoints - detect cv2.KeyPoint format
    kpts = feats['keypoints']
    if isinstance(kpts, (list, tuple)) and len(kpts) > 0 and hasattr(kpts[0], 'pt'):
        as_cv2_keypoints = True
        kpt_tensor = keypoints_to_tensor(kpts, device)
        num_keypoints = len(kpts)
    elif isinstance(kpts, np.ndarray):
        kpt_tensor = to_tensor(kpts, device=device).unsqueeze(0)
        num_keypoints = kpts.shape[0]
    else:
        kpt_tensor = kpts.to(device)
        num_keypoints = kpts.shape[1] if kpts.ndim > 1 else kpts.shape[0]
    
    debug_log(logger, "filter_feat_dict_with_mask",f"Number of keypoints: {num_keypoints}")

    # Get mask points
    mask_points = get_mask_points(mask_tensor)
    debug_log(logger, "filter_feat_dict_with_mask", f'mask_points shape: {mask_points.shape}')
        
    # Get valid indices from keypoints filtering
    kpts_unbatched = kpt_tensor.squeeze(0)
    kpts_floor = torch.floor(kpts_unbatched).long()
    kpts_ceil = torch.ceil(kpts_unbatched).long()
    valid = ~(is_in_mask(kpts_floor, mask_points.squeeze(0), logger) | 
              is_in_mask(kpts_ceil, mask_points.squeeze(0), logger))
    
    # Prepare result dictionary
    filtered_feats = {}
    
    # Apply filtering to each key in the dictionary
    for key, value in feats.items():

        # Handle keypoints specifically
        if key == 'keypoints':
            if as_cv2_keypoints:
                filtered_value = [kpts[i] for i in range(len(kpts)) if valid[i]]
            elif isinstance(value, np.ndarray):
                filtered_value = value[valid.cpu().numpy()]
            else:
                if value.ndim == 3 and value.shape[0] == 1:  # Shape [1, N, ...]
                    filtered_value = value[:, valid]
                else:  # Shape [N, ...]
                    filtered_value = value[valid]
            debug_log(logger, "filter_feat_dict_with_mask", f"Filtering key {key}.")
  
                
        # For all other items, check if they have the same length as keypoints
        else:
            matches_keypoint_dim = False
            
            if isinstance(value, (list, tuple)):
                matches_keypoint_dim = len(value) == num_keypoints
                if matches_keypoint_dim:
                    filtered_value = [value[i] for i in range(len(value)) if valid[i]]
                else:
                    filtered_value = value
                    
            elif isinstance(value, np.ndarray):
                matches_keypoint_dim = value.shape[0] == num_keypoints
                if matches_keypoint_dim:
                    filtered_value = value[valid.cpu().numpy()]
                else:
                    filtered_value = value
                    
            elif isinstance(value, torch.Tensor):
                if value.ndim > 1 and value.shape[0] == 1:
                    matches_keypoint_dim = value.shape[1] == num_keypoints
                    if matches_keypoint_dim:
                        filtered_value = value[:, valid]
                    else:
                        filtered_value = value
                else:
                    matches_keypoint_dim = value.shape[0] == num_keypoints
                    if matches_keypoint_dim:
                        filtered_value = value[valid]
                    else:
                        filtered_value = value
            else:
                # Unknown type, skip filtering
                filtered_value = value
                
            #   Log whether item was filtered based on dimensionality
            if matches_keypoint_dim:
                debug_log(logger, "filter_feat_dict_with_mask", f"Filtering key {key} as it matches keypoint dimension")
            else:
                debug_log(logger, "filter_feat_dict_with_mask", f"Skipping filtering for key {key} as it doesn't match keypoint dimension")
    
        filtered_feats[key] = filtered_value
    
    # Log results
    if logger:
        num_filtered = (len(filtered_feats['keypoints']) if isinstance(filtered_feats['keypoints'], (list, tuple)) else 
                       filtered_feats['keypoints'].shape[0] if isinstance(filtered_feats['keypoints'], np.ndarray) else 
                       filtered_feats['keypoints'].shape[1])
        debug_log(logger, "filter_feat_dict_with_mask", f"Original features count: {num_keypoints}")
        debug_log(logger, "filter_feat_dict_with_mask", f"Filtered features count: {num_filtered}")
        debug_log(logger, "filter_feat_dict_with_mask", f"Removed {num_keypoints - num_filtered} features")
        
    return filtered_feats