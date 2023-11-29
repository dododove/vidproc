import cv2
import matplotlib.pyplot as plt
# Why do Contrast Strectching? To improve visibility of details by expanding the range of intensity values.
# Useful for low contrast image
# Low contrast meaning difference b/w the darkest and brightest pixel value is low.

img = cv2.imread('Contrast_Stretching/lenna.png')
gray = img[:,:,:1]
gray = gray.squeeze()
out = gray.copy()
height, width = gray.shape
high = gray.max()
low = gray.min()

for i in range(height):
    for j in range(width):
        out[i][j] = ((gray[i][j]-low)*255 / (high-low))
cv2.imshow('original', gray)
cv2.imshow('stretching', out)

plt.figure()
plt.subplot(1,2,1)
plt.hist(gray.ravel(), 256, [0,256])
plt.subplot(1,2,2)
plt.hist(out.ravel(), 256, [0,256])
plt.show()

cv2.waitKey(0)
