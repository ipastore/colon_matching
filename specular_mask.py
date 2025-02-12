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


def process_images(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for img_path in Path(input_dir).glob('*.png'):
        # Read the image
        img = cv2.imread(str(img_path))
        # Convert to grayscale
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Create the mask
        mask = create_mask(frame_gray)
        # Apply the mask to the image
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        # Save the processed image
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), masked_img)
        print(f'Saved processed image to {output_path}')


process_images('data/easy/095', 'data/specular_masked/easy/095')
process_images('data/medium/093', 'data/specular_masked/medium/093')
process_images('data/medium/094', 'data/specular_masked/medium/094')
process_images('data/medium/095', 'data/specular_masked/medium/095')
process_images('data/hard/118', 'data/specular_masked/hard/118')