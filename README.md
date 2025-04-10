# Master's Project Repository
![image](https://github.com/user-attachments/assets/0a1690d7-3197-4d05-bd9f-3ef09538e503)

[https://www.linkedin.com/in/elijah-ki-zerbo-a00484198/recent-activity/all/
](https://www.linkedin.com/posts/elijah-ki-zerbo-a00484198_engineering-imageprocessing-ai-activity-7234454665462001665-2Zj6?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC5YGKABJ_CFfAcqdaFzM5Zayex6DFq2SaI)


## Description
This repository contains Python scripts for various image inpainting techniques implemented as part of my Master's project. The implemented techniques include Curvature-Driven Diffusions (CDD), Nonlinear Diffusion Filter (NDF), Total Variation (TV) inpainting, and optimization scripts for tuning parameters.

## Portfolio.pdf

Please take the time to read the "Final Portfolio.pdf" the sum and results of this research project of mine. 

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/kizer/Masters-project.git
   ```

2. Ensure you have the necessary dependencies installed:
   - numpy
   - scipy
   - matplotlib
   - scikit-image
   - opencv-python
   - Pillow

3. Run the scripts using a Python environment.

### CDD Inpainting
- File: `CDD.py`
- Function: `cdd_inpainting(image, mask, g, iterations=100, tau=0.1)`
- Parameters:
  - `image`: 2D numpy array representing the grayscale image.
  - `mask`: 2D boolean numpy array where True indicates missing pixels to inpaint.
  - `g`: Function that modifies diffusion based on curvature.
  - `iterations`: Number of iterations to run the inpainting process.
  - `tau`: Time step size.

### NDF Inpainting
- File: `Nonlinear_diff.py`
- Function: `nonlinearDiffusionFilter(image, iterations=5, lamb=1.0, tau=0.125)`
- Parameters:
  - `image`: Input image for inpainting.
  - `iterations`: Number of iterations for the filtering process.
  - `lamb`: Lambda parameter for diffusivity.
  - `tau`: Time step size.

### TV Inpainting
- File: `TV.py`
- Function: `TV(input_img, lambda_val, mask, T, dt)`
- Parameters:
  - `input_img`: Input image for inpainting.
  - `lambda_val`: Lambda value for the TV inpainting.
  - `mask`: Mask indicating regions to inpaint.
  - `T`: Total time for the inpainting process.
  - `dt`: Time step size.

### Optimization Scripts
- Files: `Optimization_CDD.py`, `Optimization_NDF.py`, `Optimization_TV.py`
- These scripts perform optimization to find the best parameters for inpainting techniques.

### Video Inpainting
- Files: `VIdeo-TV.py`, `Video.py`
- These scripts perform inpainting on a sequence of images.

For more detailed usage instructions, refer to the comments in the respective Python scripts.

## Project Files
The repository contains multiple Python scripts implementing various image inpainting techniques, each with specific functions and usage instructions. Kindly refer to the individual scripts for detailed information on each technique and how to use them.

---

Feel free to explore and experiment with the provided scripts for image inpainting using different techniques and optimization approaches. If you encounter any issues or have any questions, please feel free to reach out.

---

Additionally, don't forget to read the "Final Portfolio.pdf" for the summary and results of this research project.

For specific instructions on each technique and optimization script, refer to the respective Python files in the repository.
