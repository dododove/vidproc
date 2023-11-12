import cv2
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#Input Image as PIL
img = Image.open('mbv.png')

#Rotate
img = img.rotate(90)

#Save
img.save('mbv_rt90.png')