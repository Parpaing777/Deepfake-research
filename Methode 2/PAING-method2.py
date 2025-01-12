"""
Script to analyze the image using the method mentioned in the papers ref3 and ref3bis
This method utilizes the noise extracted from the image to compute the Kappa and Eigen features 
and uses them to detect tampered regions in the image.
This script only analyzes a single image.
Attention: This method does not work well in terms of detecting deepfakes.

@Author: Thant Zin Htoo PAING
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Extract RGB noise with median filter
def extract_noise(img):
    img_float = img.astype(np.float32) # convert the image to float32
    median = cv.medianBlur(img, 3).astype(np.float32) # apply median filter to the image
    # median = cv.medianBlur(img, 3)
    noise = img_float - median
    # noise = img - median
    return noise

# Split into sliding windows
def sliding_window(img, windowSize, stepSize):
    h, w, _ = img.shape
    for y in range(0, h-windowSize+1, stepSize): # loop through the height of the image
        for x in range(0, w-windowSize+1, stepSize): # loop through the width of the image
            yield (x, y, img[y:y+windowSize, x:x+windowSize])

# Compute Noise Related feature (Kappa k)
def compute_kappa(window_noise):
    reshaped_noise = window_noise.reshape(-1, 3)
    barycenter = np.mean(reshaped_noise, axis=0)
    distances = np.linalg.norm(reshaped_noise - barycenter, axis=1)
    kappa = np.var(distances)
    return kappa


# Compute eigen feature (Ck)
def compute_eigen(window_noise):
    reshape = window_noise.reshape(-1, 3) # reshape the 3D array to 2D array
    if np.var(reshape) < 1e-6: # if the value of the variance is less than 1e-6
        return 0
    # PCA
    pca = PCA(n_components=1) # instantiate PCA with 1 component
    projections = pca.fit_transform(reshape) # fit the PCA model and project the data to the principal component

    # Select the middle projection
    lower,upper = np.percentile(projections, [25, 75]) # find the 25th and 75th percentile of the projections
    selected_projections = projections[(projections >= lower)&(projections <= upper)] # select the projections that are within the 25th and 75th percentile

    # Compute the eigen feature
    eigen_feature = np.mean(selected_projections) # find the average of the selected projections
    return eigen_feature

# Analyze the image
def analyze_img(img, windowSize=16, stepSize=4):
    noise = extract_noise(img) # extract the noise from the image
    h, w, _ = img.shape
    # initialize the feature maps
    kappa_heatmap = np.zeros((h, w))
    eigen_heatmap = np.zeros((h, w))

    # loop through the sliding windows
    for x,y,window in sliding_window(noise, windowSize, stepSize):
        if np.var(window) < 1e-6: # if the variance of the window is less than 1e-6, skip the window
            continue
        kappa = compute_kappa(window) # compute the kappa feature
        eigen = compute_eigen(window) # compute the eigen feature
        kappa_heatmap[y:y+windowSize, x:x+windowSize] = kappa # store the kappa feature in the heatmap
        eigen_heatmap[y:y+windowSize, x:x+windowSize] = eigen # store the eigen feature in the heatmap

    # Show the tempered map (i.e mark the pixels as tempered if the kappa feature is greater than the threshold)
    mean_kappa = np.mean(kappa_heatmap) # find the mean of the kappa heatmap
    std_kappa = np.std(kappa_heatmap) # find the standard deviation of the kappa heatmap
    adaptive_threshold = mean_kappa + 0.5*std_kappa # calculate the adaptive threshold
    tampered_map = (kappa_heatmap > adaptive_threshold).astype(np.uint8) # create a tampered map by thresholding the kappa heatmap
    return kappa_heatmap, eigen_heatmap, tampered_map

# Visualize the results
def visual_res(org_img, kappa_heatmap, eigen_heatmap, tampered_map):
    plt.figure(figsize=(12,8))
    plt.subplot(1,4,1)
    plt.imshow(cv.cvtColor(org_img,cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(kappa_heatmap, cmap='hot')
    plt.title('Kappa Heatmap')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(eigen_heatmap, cmap='cool')
    plt.title('Eigen Heatmap')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(tampered_map, cmap='gray')
    plt.title('Tampered Map')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    img_path = r"path\to\image\you\want\to\analyze" # Change the path to the image you want to analyze
    img = cv.imread(img_path)
    kappa_heatmap, eigen_heatmap, tampered_map = analyze_img(img)
    visual_res(img, kappa_heatmap, eigen_heatmap, tampered_map)