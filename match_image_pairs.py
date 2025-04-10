from specular_mask import *
from matching.viz import plot_matches, plot_kpts_2_images
import time
from my_logging import debug_log
import gc
from memory_profiler import profile
import psutil

def match_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger = None, resize = None, masking=False, plot_kpts=False):
            """
            Process pairs of images using the given matcher and save the resulting plots.
            """
                        
            img0 = matcher.load_image(img_path0, resize=resize)
            img1 = matcher.load_image(img_path1, resize=resize)
            debug_log(logger, 'match_image_pairs', f'img0 shape: {img0.shape}')
            debug_log(logger, 'match_image_pairs', f'img1 shape: {img1.shape}')
            
            # Convert tensor images to BGR NumPy arrays (H, W, C) (row, col, channel)
            img0_np= get_bgr_image(img0)
            img1_np = get_bgr_image(img1)
            mask0 = None
            mask1 = None

            if masking:
                
                # Get mask zero points and masked images for both images
                mask0, masked_img0 = get_mask_and_masked_image(img0_np)
                mask1, masked_img1 = get_mask_and_masked_image(img1_np)
                debug_log(logger, 'match_image_pairs', f'mask0 shape: {mask0.shape}')
                debug_log(logger, 'match_image_pairs', f'mask1 shape: {mask1.shape}')
                
                # Convert the numpy mask to a torch tensor for later processing
                # Convert a NumPy mask (2D array, dtype=np.uint8) to a torch tensor
                # with shape (1, H, W) and dtype=torch.uint8.
                mask0 = torch.from_numpy(mask0).unsqueeze(0)
                mask1 = torch.from_numpy(mask1).unsqueeze(0)

            # Match the images and log time
            result = matcher(img0, img1, mask0=mask0, mask1=mask1, logger=logger)

            # Check if any matches were found after filtering
            if len(result['matched_kpts1']) == 0:
                logger.info(f'No matches found for pair: {img_path0.stem} and {img_path1.stem} using {model_name}')
                del img0, img1, mask0, mask1, img0_np, img1_np, masked_img0, masked_img1
                gc.collect()
                return  # Skip this pair
            
            # Commented because this is plotted upstream. This is an old implementation when only plotting without any E matrix computation for easy, medium and hard cases.
            # if masking:
            #     start_plotting = time.perf_counter()
            #     plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
            #     plot_matches(masked_img0, masked_img1, result, show_all_kpts=plot_kpts, save_path=plot_path)
            #     end_plotting = time.perf_counter()
            #     debug_log(logger, 'match_image_pairs', f'Plotting matches took {end_plotting - start_plotting:.3f} seconds')
            #     debug_log(logger, 'match_image_pairs', f'Saved plot to {plot_path}')
                
            # else:
            #     start_plotting = time.perf_counter()
            #     plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
            #     plot_matches(img0, img1, result, show_all_kpts=plot_kpts, save_path=plot_path)
            #     end_plotting = time.perf_counter()
            #     debug_log(logger, 'match_image_pairs', f'Plotting matches took {end_plotting - start_plotting:.3f} seconds')
            #     debug_log(logger, 'match_image_pairs', f'Saved plot to {plot_path}')
            
            del mask0, mask1, img0_np, img1_np
            gc.collect()
            return result, img0, img1, masked_img0, masked_img1
