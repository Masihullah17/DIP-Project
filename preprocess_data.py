import numpy as np
import pandas as pd
import pickle
from skimage import data, img_as_float
from skimage import exposure

# Preprocessing Functions
def contrastStretching(image):
	# Implement contrast streching here
	img_eq = exposure.equalize_hist(image)
	return img_eq

def faceEncoding(image):
	# Implement face encodings here
	encoding = None
	return encoding

def generateIntensityImages(image):
	# Implement generation of images with different intensites
	generatedImages = []
	return generatedImages

def pseudoLabelledData():
	# Loading data which has been labelled using pseudo labelling technique
	pseudoImages = []
	pseudoLabels = []
	return (pseudoImages, pesudoLabels)

if __name__ == "__main__":
	# Loading the dataset
	dataset = pd.read_csv("fer2013.csv")

	for category in dataset['Usage'].unique():
		categoryData = dataset[dataset['Usage'] == category]
		samples = categoryData['pixels'].values
		labels = categoryData['emotion'].values
		encodings = list()
		generatedImages = []
		generatedLabels = []
		generatedEncodings = []

		for i in range(len(samples)):
			try:
				image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((48, 48))
				image = contrastStretching(image)
				encodings[i] = faceEncoding(image)
				generatedImages += generateIntensityImages(image)
				generatedLabels += [labels[i]] * len(generatedImages)
				for j in range(len(generatedImages)):
					generatedEncodings[j] = faceEncoding(generatedImages[j])
			except Exception as e:
				print("Error : " + str(e))
		
		samples += generatedImages
		labels += generatedLabels
		encodings += generatedEncodings

		data = {'X' : samples, 'y' : labels, 'encodings' : encodings}
		with open(category + "_data.pickle", 'wb') as f:
			pickle.dump(data, f)
			f.close()