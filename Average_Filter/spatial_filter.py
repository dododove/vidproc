import cv2
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Open Image
img = cv2.imread('mbv.png')

#Average Filter 5x5 and 10x10
img_avg_3x3 = cv2.blur(img, (3,3))
img_avg_7x7 = cv2.blur(img, (7,7))

cv2.imwrite('mbv_avg_3x3.png', img_avg_3x3)
cv2.imwrite('mbv_avg_7x7.png', img_avg_7x7)

#Plot
fig, subs = plt.subplots(nrows=3, figsize=(5,15))
for sub, image in zip(subs.flatten(),[img, img_avg_3x3, img_avg_7x7]):
    sub.imshow(image)
plt.savefig('mbv_avg_filter.png')