import numpy as np
import pandas as pd
import pickle
from skimage import data, img_as_float
from skimage import exposure
import face_recognition
import cv2

eyesDetector = cv2.CascadeClassifier('./haarcascades/eyes.xml')
mouthDetector = cv2.CascadeClassifier('./haarcascades/mouth.xml')
noseDetector = cv2.CascadeClassifier('./haarcascades/nose.xml')

# Preprocessing Functions
def contrastStretching(image):
	# Implement contrast streching here
	img_eq = exposure.equalize_hist(image)
	return img_eq

def faceEncoding(image, model="large"):
	# Implement face encodings
	try:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		encoding = face_recognition.face_encodings(image, model=model)[0]
	except IndexError:
		encoding = np.zeros((128,))
	return encoding

def findFaces(image, model="cnn"):
	faces = face_recognition.face_locations(image, model=model)
	croppedFaceImages = []
	for face in faces:
		top, right, bottom, left = face
		croppedFace = image[top:bottom, left:right]
		croppedFaceImages.append(croppedFace)
	return croppedFaceImages

def detectCascade(face, detect):
	try:
		if detect == "mouth":
			size = (25, 15)
			x, y, w, h = mouthDetector.detectMultiScale(face)[0]
		elif detect == "eyes":
			size = (35, 16)
			x, y, w, h = eyesDetector.detectMultiScale(face)[0]
		elif detect == "nose":
			size = (25, 15)
			x, y, w, h = noseDetector.detectMultiScale(face)[0]
		else:
			return None
	except IndexError:
		return np.zeros(size[::-1])
	cropped = cv2.resize(face[y:y+h, x:x+w], size)
	return cropped

def generateIntensityImages(image):
	# Implement generation of images with different intensites
	generatedImages = []
	return generatedImages

def pseudoLabelledData():
	# Loading data which has been labelled using pseudo labelling technique
	pseudoImages = []
	pseudoLabels = []
	return (pseudoImages, pseudoLabels)

def applyPreProcessing(image, label):
	global generatedImages, generatedLabels, generatedEncodings, eyes, noses, mouths
	generatedImages.append(image)
	generatedLabels.append(label)
	generatedEncodings.append(faceEncoding(image))
	eyes.append(detectCascade(image, detect="eyes"))
	noses.append(detectCascade(image, detect="nose"))
	mouths.append(detectCascade(image, detect="mouth"))

def processImage(image, label):
	for face in findFaces(image):
		grayFace = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
		grayFace = contrastStretching(grayFace)
		applyPreProcessing(grayFace, label)

		for generated in generateIntensityImages(grayFace):
			applyPreProcessing(generated, label)

if __name__ == "__main__":
	# Loading the dataset
	dataset = pd.read_csv("./data/fer2013.csv")

	for category in dataset['Usage'].unique():
		categoryData = dataset[dataset['Usage'] == category]
		samples = categoryData['pixels'].values
		labels = categoryData['emotion'].values

		eyes = list()
		noses = list()
		mouths = list()
		generatedImages = []
		generatedLabels = []
		generatedEncodings = []

		print("Total {} {} images".format(len(samples), category))
		for i in range(len(samples)):
			try:
				image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((48, 48)).astype(np.uint8)
				image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
				processImage(image, labels[i])
				if (i+1) % 1000 == 0:
					print("Done %d images."%i)
			except Exception as e:
				print("Error : " + str(e))

		if category == "Training":
			samples, labels = pseudoLabelledData()
			print("Total %d pseudo labelled images"%len(samples))
			for i in range(len(samples)):
				try:
					processImage(samples[i], labels[i])
					if (i+1) % 1000 == 0:
						print("Done %d pseudo labelled images."%i)
				except Exception as e:
					print("Error : " + str(e))

		print({'X' : len(generatedImages), 'y' : len(generatedLabels), 'encodings' : len(generatedEncodings), 'eyes' : len(eyes), 'noses' : len(noses), 'mouths' : len(mouths)})

		data = {'X' : generatedImages, 'y' : generatedLabels, 'encodings' : generatedEncodings, 'eyes' : eyes, 'noses' : noses, 'mouths' : mouths}
		with open("./data/" + category + "_data.pickle", 'wb') as f:
			pickle.dump(data, f)
			f.close()