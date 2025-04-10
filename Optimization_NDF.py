```python
# Function to generate a mask with a small square in the middle
def generate_square_mask(image, square_size=50):
    # Calculate center coordinates of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_size = square_size // 2
    # Create a mask with a square centered in the image
    mask = np.zeros(image.shape, dtype=bool)
    mask[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size] = True
    return mask

# Parameters for optimization
iterations_values = range(1, 21)  # Vary number of iterations from 1 to 20
lamb_values = np.linspace(0.1, 2.0, 10)  # 10 values between 0.1 and 2.0
tau = 0.125  # Fixed tau

best_psnr = 0
best_mse = float('inf')
best_iterations = 0
best_lamb = 0

# Load the original image and convert it to grayscale
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
gray_image = color.rgb2gray(original)

# Generate a mask with Gaussian noise
mask = generate_gaussian_noise_mask(gray_image, noise_level=0.1, threshold_factor=1.5)

# Apply the mask to the grayscale image (set masked areas to 0)
masked_image = np.copy(gray_image)
masked_image[mask] = 0

# Iterate over different combinations of iterations and lambda values
for iterations in tqdm(iterations_values, desc='Iterations Loop'):
    for lamb in tqdm(lamb_values, desc='Lambda Iterations', leave=False):
        # Inpaint the image using nonlinear diffusion filter
        inpainted_image = nonlinearDiffusionFilter(masked_image, iterations=iterations, lamb=lamb, tau=tau)
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        psnr_value = cv2.PSNR(gray_image, inpainted_image)
        
        # Calculate MSE (Mean Squared Error)
        mse_value = mse(gray_image, inpainted_image)
        
        # Check if the current combination is better than the previous best
        if psnr_value > best_psnr or (psnr_value == best_psnr and mse_value < best_mse):
            best_psnr = psnr_value
            best_mse = mse_value
            best_iterations = iterations
            best_lamb = lamb
            print(f"New Best PSNR: {best_psnr} dB, MSE: {best_mse}, Iterations: {best_iterations}, Lambda: {best_lamb}")

# Print the best parameters found during optimization
print(f"Best Iterations: {best_iterations}, Best Lambda: {best_lamb}")
print(f"Best PSNR: {best_psnr} dB, Best MSE: {best_mse}")

# Optionally, visualize the best inpainting result
best_inpainted_image = nonlinearDiffusionFilter(masked_image, iterations=best_iterations, lamb=best_lamb, tau=tau)

# Display the masked image and the best inpainted image side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(masked_image, cmap='gray')
axes[0].set_title('Masked Image')
axes[0].axis('off')

axes[1].imshow(best_inpainted_image, cmap='gray')
axes[1].set_title('Best Inpainted Image')
axes[1].axis('off')

plt.show()
```