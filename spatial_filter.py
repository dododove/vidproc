import cv2
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#Open Image
img = Image.open('mbv.png')
#Apply Gaussian Filter
gb = transforms.GaussianBlur((5,5))
img_gb = gb(img)
#Save Image
img_gb.save('mbv_gb.png')

#Apply Sharpening Filter
img_sharp = transforms.functional.adjust_sharpness(img, 2.0)
img_sharp.save('mbv_sharp.png')