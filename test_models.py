import os
import itertools
from matching import get_matcher
from matching.utils import get_default_device
from matching.viz import plot_matches
from pathlib import Path
import matplotlib.pyplot as plt
from my_logging import setup_logging
import numpy as np

########################################################## CONFIG ##########################################################
logger = setup_logging()

models = ['sift-nn']
############################# CHOOSE DATASET #############################
dataset_type = 'wout_mask'
# dataset_type = 'specular_masked'
########################################################## CONFIG ##########################################################
image_dir = Path(f'data/{dataset_type}')

logger.info('Starting easy case')
image_dir1 = Path(f'{image_dir}/easy/095')
image_paths = list(image_dir1.glob('*.png'))
pairs = list(itertools.combinations(image_paths, 2))
pairs = pairs[:1]

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

        result = matcher(img0, img1)
        if len(result['matched_kpts1']) == 0:
            logger.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name} in easy level')
            continue
        plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
        img0 = np.clip(img0, 0, 1)
        img1 = np.clip(img1, 0, 1)
        plot_matches(img0, img1, result, save_path=plot_path)
        logger.info(f'Saved plot to {plot_path}')

