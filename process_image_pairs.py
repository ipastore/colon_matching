from specular_mask import *
from matching.viz import plot_matches


def process_image_pairs(img_path0, img_path1, output_dir, model_name, matcher, logger):
            """
            Process pairs of images using the given matcher and save the resulting plots.
            """

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
                logger.info(f'No matches found for pair: {img_path0} and {img_path1} using {model_name}')
                return  # Skip this pair

            # Filter matched keypoints using helper function
            result_filtered = filter_result(result, mask0_zero_points, mask1_zero_points)
            # logger.info(f'Deleted {len(result["matched_kpts0"]) - len(result_filtered["matched_kpts0"])} matches due to specularities')

            plot_path = output_dir / f'{img_path0.stem}_{img_path1.stem}_{model_name}.png'
            plot_matches(masked_img0, masked_img1, result_filtered, save_path=plot_path)
            logger.info(f'Saved plot to {plot_path}')