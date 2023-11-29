import cv2
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



img = cv2.imread('mbv.png')

# #Apply Sharpening Filter
# img_sharp = transforms.functional.adjust_sharpness(img, 2.0)
# img_sharp.save('mbv_sharp.png')

spf0 = np.array([[-1,-1],[-1,4]])
spf1 = np.array([[-1,-1,-1],[-1,9,1],[-1,-1,-1]])
spf2 =np.array([[-1,-1,-1,-1],[-1,3,3,-1],[-1,3,4,-1],[-1,-1,-1,-1]])

out0 = cv2.filter2D(img, -1, spf0)
out1 = cv2.filter2D(img, -1, spf1)
out2 = cv2.filter2D(img, -1, spf2)

cv2.imwrite('mbv_sharpen_2x2c.png', out0)
cv2.imwrite('mbv_sharpen_3x3c.png', out1)
cv2.imwrite('mbv_sharpen_4x4c.png', out2)