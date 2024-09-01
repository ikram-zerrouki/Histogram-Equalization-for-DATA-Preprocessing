import os
import numpy as np
import cv2

# Directories for input and output images
dataset_dir = os.path.join(os.path.dirname(__file__), 'Dataset subfolder1', 'Dataset subfolder2')
equalized_dir = os.path.join(os.path.dirname(__file__), 'Dataset subfolder1 equalized', 'Dataset subfolder2 equalized')

# Create directories to save processed images if they don't exist
if not os.path.exists(equalized_dir):
    os.makedirs(equalized_dir)

# Function to preprocess the image (resize, histogram equalization, standardization)
def histogram_equalization(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

        # Convert to grayscale if not already
        if len(image.shape) == 3:  # If image has 3 channels (BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(image)

        return equalized_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Loop over images in dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.png'):  # Adjust according to file format
        image_path = os.path.join(dataset_dir, filename)
        equalized_image = histogram_equalization(image_path)

        if equalized_image is not None:
            # Save the equalized image to the new directory
            equalized_image_path = os.path.join(equalized_dir, filename)
            cv2.imwrite(equalized_image_path, equalized_image)

print("Histogram equalization completed and images saved.")
