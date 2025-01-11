"""
This script is for detecting faces in images, cropping them and saving them in a new directory

@Author: Thant Zin Htoo PAING
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Load the cascade from local storage or from the web
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Loading from local storage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Loading from the web

# Define image directory
img_path = r"path\to\image\directory" # Change the path to your image directory

# Create a new directory to save the images
new_dir = os.path.join(img_path, 'faces')
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

def face_off(img_path, new_dir):
    # create a list and append all images that ends with .jpg from the image directory
    img_list = []
    for file in os.listdir(img_path):
        if file.endswith('.jpg'):
            full_path = os.path.join(img_path, file)
            img_list.append(full_path)


    # Read images from the list as BGR image
    for file in img_list:
        img = cv2.imread(file)

        if img is None:
            print('Could not read the image')
            continue

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        # Add margin for cropping
        margin = 0.2 # 20% margin
        # Crop with margin
        for i,(x,y,w,h) in enumerate(face):
            xM = int(w*margin)
            yM = int(h*margin)

            x_start = max(0, x-xM)
            y_start = max(0, y-yM)
            x_end = min(img.shape[1], x+w+xM)
            y_end = min(img.shape[0], y+h+yM)

            face = img[y_start:y_end, x_start:x_end]

            # Save the images in the new directory
            cv2.imwrite(os.path.join(new_dir, f'face_{os.path.basename(file).split(".")[0]}_{i}.jpg'), face)
        

if __name__ == '__main__':
    face_off(img_path, new_dir)
    print('Process done!, Do not froget to delete false positives!')





