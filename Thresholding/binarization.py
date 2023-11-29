import cv2

img = cv2.imread('Thresholding/lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('original', img)
cv2.imshow('binary', dst)
cv2.waitKey(0)