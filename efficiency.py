import numpy as np

def mse(imageA, imageB):
    # Calculate Mean Squared Error (MSE) between two images
    # MSE is the sum of the squared difference between the two images divided by the total number of pixels
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err


#CHANGE TO JUST THE MASK 
def psnr(original, restored):
    # Calculate Peak Signal-to-Noise Ratio (PSNR) between original and restored images
    mse_value = mse(original, restored)
    if mse_value == 0:
        # If MSE is 0, it means no error, return maximum value
        return 100
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value

from skimage import io