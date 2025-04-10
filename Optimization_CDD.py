```python
# Function to generate a mask with a small square in the middle
def generate_square_mask(image, square_size=50):
    # Calculate the center coordinates of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_size = square_size // 2
    # Create a mask with a square region in the middle
    mask = np.zeros(image.shape, dtype=bool)
    mask[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size] = True
    return mask

# Parameters for optimization
tau_values = np.linspace(0.001, 0.1, 100)  # 100 values between 0.001 and 0.1
p_values = range(1, 11)  # Integer values from 1 to 10

best_psnr = 0
best_mse = float('inf')
best_tau = 0
best_p = 0

# Load the original image and convert it to grayscale
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
gray_image = color.rgb2gray(original)

# Generate a mask with a small square in the middle
mask = generate_square_mask(gray_image, square_size=20)

# Apply the mask to the grayscale image (set masked areas to 0)
masked_image = np.copy(gray_image)
masked_image[mask] = 0

# Iterate over tau and p values to find the best combination
for tau in tqdm(tau_values, desc='Tau Iterations'):
    for p in tqdm(p_values, desc='P Iterations', leave=False):
        # Define the g function based on curvature
        g = lambda s: s**p
        
        # Inpaint the image using CDD algorithm
        inpainted_image = cdd_inpainting(masked_image, mask, g, iterations=2500, tau=tau)
        
        # Calculate PSNR between original and inpainted image
        psnr_value = cv2.PSNR(gray_image, inpainted_image)
        
        # Calculate MSE between original and inpainted image
        mse_value = mse(gray_image, inpainted_image)
        
        # Check if current combination is better than the previous best
        if psnr_value > best_psnr or (psnr_value == best_psnr and mse_value < best_mse):
            best_psnr = psnr_value
            best_mse = mse_value
            best_tau = tau
            best_p = p
            print(f"New Best PSNR: {best_psnr} dB, MSE: {best_mse}, Tau: {best_tau}, P: {best_p}")

# Print the best parameters found
print(f"Best Tau: {best_tau}, Best P: {best_p}")
print(f"Best PSNR: {best_psnr} dB, Best MSE: {best_mse}")

# Generate the best inpainted image using the optimal parameters
g = lambda s: s**best_p
best_inpainted_image = cdd_inpainting(masked_image, mask, g, iterations=5000, tau=best_tau)

# Display original grayscale image and best inpainted image
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(masked_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[2].imshow(best_inpainted_image, cmap='gray')
axes[2].set_title('Best Inpainted Image')
axes[2].axis('off')

plt.show()
```