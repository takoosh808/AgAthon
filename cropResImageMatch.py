import cv2
import numpy as np

# Takes in 2 image file paths and Computes the Intersection over Union (IoU) score between two images
def compute_iou(image_path1, image_path2):

    # Load images in grayscale
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise ValueError("First image could not be loaded.")
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    if img2 is None:
        raise ValueError("Second image could not be loaded.")
    
    # Ensure images are the same size
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # Convert to binary (assuming 255 as foreground and 0 as background)
    print("Converting to binary")
    _, binary1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    
    # Compute intersection and union
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    
    # Compute IoU score as a percentage
    iou_score = (intersection / union) * 100 if union > 0 else 0
    
    return iou_score

# Example usage
if __name__ == "__main__":
    image1 = "images\Zak-W-winterBarley_1m_20220401\IMG_0937\IMG_0937_res_part01.tif"
    image2 = "images\Zak-W-winterBarley_1m_20220401\IMG_0937\IMG_0937_res_part02.tif"
    
    try:
        iou = compute_iou(image1, image2)
        print(f"IoU Score: {iou:.2f}%")
    except ValueError as e:
        print("Error:", e)