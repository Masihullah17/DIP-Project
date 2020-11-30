from preprocess_data import contrastStretching, faceEncoding, generateIntensityImages, pseudoLabelledData
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

image = cv2.imread('test_image.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Replace function with the one, you are working on
processedImage = contrastStretching(image)

# Left image is original image and right image is the processed image
plt.imshow(np.concatenate((image, processedImage), axis=1), cmap=cm.gray, vmin=0, vmax=255)
plt.show()