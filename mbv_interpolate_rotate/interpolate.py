import cv2
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#Input Image & Convert to Numpy Grayscale
img = PIL.Image.open('mbv.png')
img = np.asarray(img, dtype=float)
img = img[:,:,:1]
img = np.pad(img, (((1,1),(1,1),(0,0))), 'constant', constant_values=0.0)
# print('Img:', img)
# print('Img ndim: ', img.ndim)

#Shape of img & Save img
height, width, channel = img.shape
print('Img Shape: ', img.shape)
# cv2.imwrite('mbv_gray.png', img)

#Create 2 times larger Blank Array and apply Padding
new_arr = np.zeros((2*height, 2*width, channel), dtype=np.float32)
new_arr = np.pad(new_arr, (((1,1),(1,1),(0,0))), 'constant', constant_values=0.0)

#Fill new_arr
for c in range(channel):
    for w in range(width-1):
        for h in range(height-1):
            A,B,C,D = img[h][w][c], img[h][w+1][c], img[h+1][w][c], img[h+1][w+1][c]
            new_arr[2*h][2*w][c] = A
            new_arr[2*h][2*w+1][c] = float((A+B) / 2)
            new_arr[2*h+1][2*w][c] = float((A+C) / 2)
            new_arr[2*h+1][2*w+1][c] = float((A+B+C+D) / 4)

#Resize new_arr to 600*600 
# new_arr = np.uint8(np.clip(np.round(new_arr), 0, 255))
new_arr = new_arr[:600, :600, :]

#Show Img 2x and its shape
# print('Img 2x: ', new_arr)
# print('Img 2x Shape: ', new_arr.shape)
# print(new_arr.ndim)

#Save new_arr (Img 2x)
cv2.imwrite('mbv_2x_float.png', new_arr)
