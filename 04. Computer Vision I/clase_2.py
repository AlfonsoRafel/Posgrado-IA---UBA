import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def white_patch(img):
    max_channels = (255/np.array([np.amax(img[:, :, 0]), np.amax(img[:, :, 1]), np.amax(img[:, :, 2])]))
    for i in range(3):
        img[:, :, i] = np.round(img[:, :, i] * max_channels[i]).astype(np.uint8)

def crom_cord(img):
    sum_vec = np.sum(img, axis=2)
    img = img.astype(np.float32)
    for i in range(3):
        img[:, :, i] = img[:, :, i] / sum_vec
    return img





if __name__ == "__main__":
    folder = 'Images/WhitePatch/'
    for filename in os.listdir(folder):
        img_ori = cv.imread(os.path.join(folder, filename))
        img = img_ori.copy()
        #img = cv.medianBlur(img, 5)
        white_patch(img)
        img_ori_BGR = cv.cvtColor(img_ori, cv.COLOR_BGR2RGB)
        img_BGR = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        fig = plt.figure(figsize=[16, 8])
        fig.add_subplot(1, 2, 1)
        img = plt.imshow(img_ori_BGR)
        fig.add_subplot(1, 2, 2)
        img = plt.imshow(img_BGR)
        plt.show()

    folder = 'Images/Crom_cord/'
    for filename in os.listdir(folder):
        img_ori = cv.imread(os.path.join(folder, filename))
        img = img_ori.copy()
        img = crom_cord(img)
        img_ori_BGR = cv.cvtColor(img_ori, cv.COLOR_BGR2RGB)
        img_BGR = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        fig = plt.figure(figsize=[16, 8])
        fig.add_subplot(1, 2, 1)
        img = plt.imshow(img_ori_BGR)
        fig.add_subplot(1, 2, 2)
        img = plt.imshow(img_BGR)
        plt.show()

