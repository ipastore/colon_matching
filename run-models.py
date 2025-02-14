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
############################# CHOOSE LEVELS #############################
# levels = ['easy','medium','hard']
# levels = ['medium','hard']
levels = ['easy']
############################# CHOOSE SUBMAPS #############################
submaps_medium = [('093', '094'),('093', '095'), ('094', '095')]
submaps_hard = [('118', '093'), ('118', '094'), ('118', '095')]
############################# CHOOSE MODELS #############################
# models = ['superpoint-lg', 'sift-lg','tiny-roma', 'sift-nn']
# models = ['superpoint-lg', 'sift-lg']
models = ['sift-nn']
############################# CHOOSE DATASET #############################
# dataset_type = 'wout_mask'
dataset_type = 'specular_masked'
########################################################## CONFIG ##########################################################
image_dir = Path(f'data/{dataset_type}')


# Easy case
if 'easy' in levels:
    logger.info('Starting easy case')
    image_dir1 = Path(f'{image_dir}/easy/095')
    image_paths = list(image_dir1.glob('*.png'))
    pairs = list(itertools.combinations(image_paths, 2))

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

# Medium case
if 'medium' in levels:
    logger.info('Starting medium case')

    for submap1, submap2 in submaps_medium:
        logger.info(f'Starting submap pair: {submap1} and {submap2} for medium case')
        image_dir1 = Path(f'{image_dir}/medium/{submap1}')
        image_dir2 = Path(f'{image_dir}/medium/{submap2}')
        image_paths1 = list(image_dir1.glob('*.png'))
        image_paths2 = list(image_dir2.glob('*.png'))
        pairs = list(itertools.product(image_paths1, image_paths2))

        for model_name in models:
            logger.info(f'Starting model: {model_name} for submap pair: {submap1} and {submap2}')
            # Create output directory for the model
            output_dir = Path(f'output/medium/{submap1}_{submap2}/{model_name}')
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
                    logger.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name} in medium level with {submap1} and {submap2}')
                    continue
                plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
                img0 = np.clip(img0, 0, 1)
                img1 = np.clip(img1, 0, 1)
                plot_matches(img0, img1, result, save_path=plot_path)
                logger.info(f'Saved plot to {plot_path}')

# Hard case
if 'hard' in levels:
    logger.info('Starting hard case')

    for submap1, submap2 in submaps_hard:
        logger.info(f'Starting submap pair: {submap1} and {submap2} for hard case')
        image_dir1 = Path(f'{image_dir}/hard/{submap1}')
        image_dir2 = Path(f'{image_dir}/medium/{submap2}')
        image_paths1 = list(image_dir1.glob('*.png'))
        image_paths2 = list(image_dir2.glob('*.png'))
        pairs = list(itertools.product(image_paths1, image_paths2))

        for model_name in models:
            logger.info(f'Starting model: {model_name} for submap pair: {submap1} and {submap2}')
            # Create output directory for the model
            output_dir = Path(f'output/hard/{submap1}_{submap2}/{model_name}')
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
                    logger.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name} in hard level with {submap1} and {submap2}')
                    continue
                plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
                img0 = np.clip(img0, 0, 1)
                img1 = np.clip(img1, 0, 1)    
                plot_matches(img0, img1, result, save_path=plot_path)
                logger.info(f'Saved plot to {plot_path}')
