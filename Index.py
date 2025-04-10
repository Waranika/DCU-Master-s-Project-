from Nonlinear_diff import *
from CDD import *
from efficiency import *
from PIL import Image
from skimage import color
from skimage.color import rgb2gray
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from TV import *
from skimage.filters import gaussian
import cv2

# Function to generate a mask with a small square in the middle
def generate_square_mask(image, square_size=50):
    mask = np.zeros(image.shape, dtype=bool)
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_size = square_size // 2
    mask[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size] = True
    return mask

# Function to generate a mask with Gaussian noise exceeding a threshold
def generate_gaussian_noise_mask(image, noise_level=0.1, threshold_factor=1.0):
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, image.shape)
    
    # Calculate threshold based on noise level and factor
    threshold = noise_level * threshold_factor
    
    # Create a boolean mask where the noise exceeds the threshold
    mask = np.abs(noise) > threshold
    
    return mask

# Function to generate a mask with random pixels set to 0
def generate_random_pixel_mask(image, mask_ratio=0.1):
    # Create a mask with the same shape as the image, initialized to False
    mask = np.zeros(image.shape, dtype=bool)
    # Calculate the number of pixels to be masked
    num_pixels = image.size
    num_masked_pixels = int(num_pixels * mask_ratio)
    # Randomly choose indices to set to 0
    indices = np.random.choice(num_pixels, num_masked_pixels, replace=False)
    # Set the chosen indices in the mask to True
    mask.flat[indices] = True
    return mask

# Load the images
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
gray_image = color.rgb2gray(original)

# Generate a mask with a small square in the middle
mask = generate_square_mask(gray_image, square_size=50)

# Apply the mask to the grayscale image (set masked areas to 0)
masked_image = np.copy(gray_image)
masked_image[mask] = 0

# Apply the mask (highlight masked areas in red for visibility)
gray_image_rgb = np.stack([gray_image]*3, axis=-1)
gray_image_rgb[mask] = [1, 0, 0]  # Red color for masked areas
print("mse = ", mse(gray_image, masked_image))
print(cv2.PSNR(gray_image, masked_image))