import cv2
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Open Image
img = Image.open('mbv.png')
img = np.array(img)
img = img[:,:,:1]
img = img.squeeze()
img = img[170:170+128,:128]
print(img.shape)

plt.subplot(2,2,1)
plt.hist(img)

#Apply Gaussian Filter
for sigma in range(1,4):
    out = cv2.GaussianBlur(img, (0,0), sigma)
    cv2.imwrite(f'Gaussian_Blur/mbv_0{sigma}.png', out)
    plt.subplot(2, 2, sigma+1)
    plt.hist(out)
    
plt.savefig('plt_mbv_noiseHist.png')