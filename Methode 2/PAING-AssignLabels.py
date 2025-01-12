"""
This is a program to assign labels to the images and use the model to train the dataset.

@Author: Thant Zin Htoo PAING
"""
import cv2
from os import listdir
import pandas as pd

img_list = []

def load_img(image):
    img = cv2.imread(image)
    return img

def add_img(directory):
    """
    function to add images to the list
    """
    for filename in listdir(directory):
        if not filename.endswith(".jpg"):
            continue
        path = directory+ '/' + filename
        img_list.append(load_img(path))


add_img(r"path\to\fake\images") # Change the path to your image directory
add_img(r"path\to\real\images") # Change the path to your image directory
labels = ["Fake"]*26 + ["Real"]*25 # Replace the number with the number of images in the respective directories

df_images = pd.DataFrame()
df_images['Images'] = img_list
df_images['Label'] = labels

# Save the dataframe to a csv file
df_images.to_csv('images.csv', index=False)