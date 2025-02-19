import os
import itertools
from matching import get_matcher
from matching.utils import get_default_device
from matching.viz import plot_matches
from pathlib import Path
import matplotlib.pyplot as plt
from my_logging import setup_logging
import numpy as np
import cv2
from pathlib import PosixPath
from specular_mask import *  # imports create_mask_normalized and is_in_mask


########################################################## CONFIG ##########################################################
logger = setup_logging()

models = ['sift-nn']

########################################################## CONFIG ##########################################################
image_dir = Path(f'data')

logger.info('Starting easy case')
image_dir1 = Path(f'{image_dir}/easy/095')
image_paths = list(image_dir1.glob('*.png'))
pairs = list(itertools.combinations(image_paths, 2))
# pairs = pairs[:1]
# pairs = f_0064790_f_0064786
# pairs = [(PosixPath('data/wout_mask/easy/095/f_0064926.png'), PosixPath('data/wout_mask/easy/095/f_0064790.png'))]

for model_name in models:
    logger.info(f'Starting model: {model_name} for easy case')
    # Create output directory for the model
    output_dir = Path(f'output/easy/{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the matcher
    device = get_default_device()
    matcher = get_matcher(model_name, device=device)

    # Process each pair of images
    for img_path0, img_path1 in pairs:
        logger.info(f"Processing pair: {img_path0} and {img_path1} using {model_name}")
        img0 = matcher.load_image(img_path0, resize=512)
        img1 = matcher.load_image(img_path1, resize=512)

        # Convert tensor images to BGR NumPy arrays
        img0_np = get_bgr_image(img0)
        img1_np = get_bgr_image(img1)

        # Get mask zero points and masked images for both images
        mask0_zero_points, masked_img0 = get_mask_points_and_masked(img0_np)
        mask1_zero_points, masked_img1 = get_mask_points_and_masked(img1_np)

        # Match the images        
        result = matcher(img0, img1)
        if len(result['matched_kpts1']) == 0:
            logger.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name} in easy level')
            continue

        # Filter matched keypoints using helper function
        result_filtered = filter_result(result, mask0_zero_points, mask1_zero_points)
        logger.info(f'Deleted {len(result["matched_kpts0"]) - len(result_filtered["matched_kpts0"])} matches due to specularities')

        # Plot and log the results
        plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
        
        plot_matches(masked_img0, masked_img1, result_filtered, save_path=plot_path)

        logger.info(f'Saved plot to {plot_path}')




