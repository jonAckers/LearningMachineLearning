import numpy as np
import os
import cv2
from random import shuffle
import pickle

# Constants for the dataset
DATADIR = './flower_photos'
CATEGORIES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
IMGSIZE = 64


# Load the images for training
def createTrainingData():
    trainingData = []

    # Get images by category
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        classNum = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                # Read image
                imgArray = cv2.imread(os.path.join(path, img))
                newArray = cv2.resize(imgArray, (IMGSIZE, IMGSIZE))
                trainingData.append([newArray, classNum])

            except Exception:
                print('Image Failed!')

    # Shuffle the data so it is a different order each time runs
    shuffle(trainingData)

    return trainingData


# Split training data in features and labels
def createFeaturesAndLabels(trainingData):
    X = []
    y = []

    # Split the data
    for features, label in trainingData:
        X.append(features)
        y.append(label)

    # Reshape arrays
    targets = np.array([y]).reshape(-1)
    y = np.eye(5)[targets]
    X = np.array(X).reshape(-1, IMGSIZE)

    return X/255.0, y


# Save data so it can be used repeatedly
def createPickles(X, y):
    # Save features
    pickleOut = open('flowersX.pickle', 'wb')
    pickle.dump(X, pickleOut)
    pickleOut.close()

    # Save labels
    pickleOut = open('flowersY.pickle', 'wb')
    pickle.dump(y, pickleOut)
    pickleOut.close()


# Gets flower data
def getData():
    trainingData = createTrainingData()
    X, y = createFeaturesAndLabels(trainingData)
    createPickles(X, y)


if __name__ == '__main__':
    getData()
