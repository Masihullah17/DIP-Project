from preprocess_data import contrastStretching, faceEncoding, generateIntensityImages, pseudoLabelledData, findFaces, detectCascade
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import face_recognition

image = cv2.imread('test_image.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Replace function with the one, you are working on
# processedImage = contrastStretching(image)

processedImage = findFaces(image)
gray = cv2.cvtColor(processedImage[0], cv2.COLOR_RGB2GRAY)

print(faceEncoding(gray, model="large"))

detected = detectCascade(gray, "mouth")

plt.imshow(gray)
plt.show()

plt.imshow(detected)
plt.show()

# Left image is original image and right image is the processed image
# plt.imshow(np.concatenate((image, processedImage), axis=1), cmap=cm.gray, vmin=0, vmax=255)
# plt.show()