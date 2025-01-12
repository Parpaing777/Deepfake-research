"""
This script is the same as PAING-method2.py but it is modified to analyze multiple images and save the features in a CSV file.
This script is intented to use with another script for training a machine learning model.

@Author: Thant Zin Htoo PAING
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from os import listdir
import pandas as pd

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

# Extract numerical features 
def extract_features(kappa_heatmap,eigen_heatmap,tampered_map):
    # Compute the mean and standard deviation of the kappa heatmap
    mean_kappa = np.mean(kappa_heatmap)
    std_kappa = np.std(kappa_heatmap)
    # Compute the mean and standard deviation of the eigen heatmap
    mean_eigen = np.mean(eigen_heatmap)
    std_eigen = np.std(eigen_heatmap)
    # Compute the percentage of tampered pixels
    tampered_pixels = np.sum(tampered_map)
    total_pixels = tampered_map.size
    tampered_percentage = tampered_pixels / total_pixels
    return mean_kappa, std_kappa, mean_eigen, std_eigen, tampered_percentage

def analyze_image_features(img):
    kappa_heatmap, eigen_heatmap, tampered_map = analyze_img(img)
    features = extract_features(kappa_heatmap, eigen_heatmap, tampered_map)
    return features

if __name__ == "__main__":
    img_path_fake = r"path\to\fake\images" # Do not forget to change the path to your own path
    img_path_real = r"path\to\real\images" # Do not forget to change the path to your own path
    img_list = []
    for filename in listdir(img_path_fake):
        if not filename.endswith(".jpg"):
            continue
        path = img_path_fake + '/' + filename
        img = cv.imread(path)
        img_list.append(img)
    
    for filename in listdir(img_path_real):
        if not filename.endswith(".jpg"):
            continue
        path = img_path_real + '/' + filename
        img = cv.imread(path)
        img_list.append(img)
    
    features_list = []
    for img in img_list:
        features = analyze_image_features(img)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list, columns=['Mean Kappa', 'Std Kappa', 'Mean Eigen', 'Std Eigen', 'Tampered Percentage'])
    features_df.to_csv('features.csv', index=False)
