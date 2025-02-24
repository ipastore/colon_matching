from specular_mask import *
from matching.viz import plot_matches
import time
import matplotlib
# matplotlib.use('TkAgg')  # Set interactive backend
matplotlib.use('MacOSX')  # Set non-interactive backend
import matplotlib.pyplot as plt

#TODO: add mask0 and mask1 as parameters. Default none.
def process_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger, specular_mask=False):
            """
            Process pairs of images using the given matcher and save the resulting plots.
            """
            logger.info(f"Processing pair: {img_path0} and {img_path1} using {model_name}")
            img0 = matcher.load_image(img_path0)
            img1 = matcher.load_image(img_path1)

            # Convert tensor images to BGR NumPy arrays (H, W, C) (row, col, channel)
            img0_np= get_bgr_image(img0)
            img1_np = get_bgr_image(img1)
            mask0 = None
            mask1 = None


            if specular_mask:
                # Get mask zero points and masked images for both images
                mask0, masked_img0 = get_mask_and_masked_image(img0_np)
                mask1, masked_img1 = get_mask_and_masked_image(img1_np)

                # Convert the numpy mask to a torch tensor for later processing
                mask0= numpy_mask_to_tensor(mask0)
                mask1 = numpy_mask_to_tensor(mask1)

            # Match the images and log time
            start = time.perf_counter()
            result = matcher(img0, img1, mask0=mask0, mask1=mask1)
            end = time.perf_counter()
            logger.info(f'Matching took {end - start:.3f} seconds')

            # Check if any matches were found after filtering
            if len(result['matched_kpts1']) == 0:
                logger.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name}')
                return  # Skip this pair
            
            plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
            plot_matches(masked_img0, masked_img1, result, save_path=plot_path)
            logger.info(f'Saved plot to {plot_path}')