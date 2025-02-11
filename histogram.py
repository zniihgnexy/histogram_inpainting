import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def compute_histogram(image_path, save=True):
    """
    Compute and save the histogram of an image.
    
    Args:
        image_path (str): Path to the image file.
        save (bool): Whether to save the histogram figure.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error reading {image_path}")
        return
    
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # Plot histogram
    plt.figure()
    plt.plot(hist, color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Save the histogram figure
    if save:
        save_path = image_path.replace('.', '_histogram.')
        plt.savefig(save_path)
        print(f"Saved histogram to {save_path}")
    
    plt.close()

def process_dataset(image_root):
    """
    Process all images in a dataset directory and generate histograms.
    
    Args:
        image_root (str): Root directory containing image files.
    """
    image_paths = glob(os.path.join(image_root, '*.*'))
    for img_path in image_paths:
        compute_histogram(img_path)

# Example usage
if __name__ == "__main__":
    dataset_path = "raw_data/diffusion_edge/data/image/raw"  # Adjust as necessary
    process_dataset(dataset_path)
