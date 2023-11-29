import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(img):
    # Shape of image
    N,M = img.shape
    # G us gray scale, H is Input Histogram
    G = 256
    H = np.zeros(G)
    # Record frequence of such pixel value of an img
    for g in img.ravel():
        H[g] += 1
    # g_min is the lowest occuring gray level in the image
    g_min = np.min(np.nonzero(H))
    
    # H_c is Cumulative image hist
    H_c = np.zeros_like(H)
    H_c[0] = H[0]
    for g in range(1, G):
        H_c[g] = H_c[g-1] + H[g]
    H_min = H_c[g_min]
    
    # T is Transformation array
    T = np.round( (H_c-H_min) / (M*N-H_min)*(G-1))
    
    # Fill equalized output img 
    out = np.zeros_like(img)
    for n in range(N):
        for m in range(M):
            out[n,m] = T[img[n,m]]
    
    return out, T

# Open Image
img = cv2.imread('mbv.png')

# Make input grayscale
img = img[:,:,:1]
img = img.squeeze()
out_n, T = histogram_equalization(img)
print(T.shape)

# Save equalized image
cv2.imwrite('mbv_grayscale.png', img)
cv2.imwrite('mbv_histEqual.png', out_n)

# Plot 
plt.hist(img)
plt.savefig('plt_mbv_hist.png')

plt.hist(out_n)
plt.savefig('plt_mbv_equalHist.png')