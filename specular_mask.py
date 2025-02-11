import cv2
import numpy as np


def create_mask(self,frame_gray):
        # Create a mask to avoid detection in specularities in the image (bright spots)
        kernel = np.ones((5,5),np.uint8)
        thresh_bright = cv2.threshold(frame_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
        thresh_bright = cv2.erode(thresh_bright, kernel, iterations=5)
        # thresh = cv2.dilate(thresh, kernel, iterations=5)
        # Create a mask to avoid detection in dark spots
        thresh_dark = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_BINARY)[1]
        thresh_dark = cv2.erode(thresh_dark, kernel, iterations=3)
        # Combine both masks
        thresh = cv2.bitwise_and(thresh_bright, thresh_dark)
        return thresh

