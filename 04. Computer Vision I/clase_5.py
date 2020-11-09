import cv2 as cv
import numpy as np
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

img_list = []
lbp_list = []
if __name__ == "__main__":
    folder = 'Images/LBP/one/'
    radius = 1
    n_points = 8 * radius
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method='default')
        img_list.append(gray)
        lbp_list.append(lbp)

        # the histogram of the data
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()
    i=0
    j=0
    for ax in ax_img:
        ax.imshow(img_list[i])
        i = i+1
    for ax in ax_hist:
        n, bins, patches = ax.hist(lbp_list[j].ravel(), bins=20, density=True)
        j = j+1
    plt.show()

