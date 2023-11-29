import cv2
import numpy as np

# Why do we use gamma correction? Human perception of brightness is not linear.
# Our eyes are more sensitive to changes in darker regions.
# Gamma correction helps compensate this non-linearity in human perception.
# O=I^r where r is gamma value, I,O is input, output.

# Gamma Value g(2.2 is commonly used for standard display)
g = 2.2

# Input Image img
img = cv2.imread('Gamma_Correction/lenna.png')
# Output Image out
out = img.copy()
# cv2 is int so covert to float for processing
out = img.astype(float)
# Why 1/g? Higher g represents higher brightness
out = ((out/255)**(1/g))*255
# Convert back to int8 to match cv2 param
out = out.astype(np.uint8)
# Show I and O
cv2.imshow('original', img)
cv2.imshow('gamma', out)
# To prevent image disappearing too quick
cv2.waitKey(0)