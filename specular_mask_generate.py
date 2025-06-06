from PIL import Image
from pathlib import Path
import torchvision.transforms as tfm
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    
    img_dir = Path('./data/seq_001/sub_maps_images/6')
    output_dir = img_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True) 

    for img_path in img_dir.glob('*.png'):

        img_tensor = load_image(img_path)
        img_np = get_bgr_image(img_tensor)
        mask, _ = get_mask_and_masked_image(img_np)
        # visualize_mask_and_image(mask, masked_img)
        colmap_mask = adapt_mask_to_colmap(mask)
        mask_path = output_dir / f"{img_path.stem}.png.png"
        cv2.imwrite(str(mask_path), colmap_mask)

    print(f"All masked images saved to {output_dir}")

    return

def load_image(path: str | Path) -> torch.Tensor:

    pil_img = Image.open(path).convert("RGB")  # Load image
    img = tfm.ToTensor()(pil_img)
    del pil_img  

    return img

def visualize_mask_and_image(mask, masked_img):
    """
    Visualizes the mask and the masked image.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(masked_img)
    plt.title('Masked Image')
    plt.axis('off')
    
    plt.show()

def create_mask_normalized(frame_gray_norm):
    """
    Create a mask for specularities for images normalized to [0,1].
    The bright threshold 220/255 (~0.86) is inverted and the dark threshold 10/255 (~0.04) is applied.
    0 = Specularities and dark spots, 1 = everything else.

    """
    kernel = np.ones((5, 5), np.float32)
    # Threshold for bright spots (inverted)
    thresh_bright = cv2.threshold(frame_gray_norm, 220/255.0, 1, cv2.THRESH_BINARY_INV)[1]
    thresh_bright = cv2.erode(thresh_bright, kernel, iterations=10)
    # Threshold for dark spots
    thresh_dark = cv2.threshold(frame_gray_norm, 10/255.0, 1, cv2.THRESH_BINARY)[1]
    thresh_dark = cv2.erode(thresh_dark, kernel, iterations=10)
    # Combine both masks
    thresh = cv2.bitwise_and(thresh_bright, thresh_dark)
    # Convert mask values from {0,1} to {0,255} as uint8
    return thresh.astype(np.uint8)

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

def get_bgr_image(img):
    """Convert a PyTorch tensor image (C,H,W) to a NumPy BGR image while freeing memory."""
    img_np = cv2.cvtColor(img.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
    
    del img
    return img_np


def adapt_mask_to_colmap(mask):
    """
    Convert a mask from {0,1} to {0,255} as uint8 for COLMAP compatibility.
    """
    return (mask * 255).astype(np.uint8)



if __name__ == "__main__":
    main()
