import cv2
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from random import random

def salt_and_pepper(image,p):
    output=np.zeros(image.shape, np.uint8)
    thres = 1-p
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()
            if rdn<p:
                output[i][j] = 0
            elif rdn>thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def median_filter(img, filter_size=(3,3), stride=1):
    img_shape = np.shape(img)
    result_shape = tuple(np.int64((np.array(img_shape)-np.array(filter_size))/stride+1))
    result = np.zeros(result_shape)
    for h in range(0, result_shape[0], stride):
        for w in range(0, result_shape[1], stride):
            tmp = img[h:h+filter_size[0], w:w+filter_size[1]]
            tmp = np.sort(tmp.ravel())
            result[h,w] = tmp[int(filter_size[0]*filter_size[1]/2)]
    return result

#Input and make grayscale
img = cv2.imread('lenna480.png')
img = img[:,:,:1]
img = img.squeeze()

#Apply noise
img = salt_and_pepper(img, 0.2)
cv2.imwrite('lenna_noise.png',img)

#Apply Median Filter 
out = median_filter(img, (3,3), 1)
cv2.imwrite('lenna_median.png', out)


            