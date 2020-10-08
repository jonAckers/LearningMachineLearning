from FlowerClassifier import getFlowerData
import time
import pickle
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


# Load the data
def getData():
    try:
        # If the data has already been generated use that
        X, y = readPickles()
    except Exception:
        # Load new data using getFlowerData script
        getFlowerData.getData()
        X, y = readPickles()

    return X, y


# Load the data from pickle files
def readPickles():
    X = pickle.load(open('flowersX.pickle', 'rb'))
    y = pickle.load(open('flowersY.pickle', 'rb'))

    return X, y


# Build neural model
def createNeuralNet(X):
    # Add layers
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(5, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# Train model on data
def trainModel(model, X, y, tensorBoard):
    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorBoard])


# Test the model once it has been trained
def testModel(model, X, y):
    CATEGORIES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    # Select 10 random images
    indices = sample(range(len(X)), 10)
    xTest = X[indices]
    yTest = y[indices]

    # Make prediction for each image
    predictions = model.predict(xTest)

    # Display results
    print()
    for i in range(len(xTest)):
        p = CATEGORIES[np.argmax(predictions[i])]
        a = CATEGORIES[np.where(yTest[i] == 1)[0][0]]

        print('Prediction: %10s     |     Actual: %10s' % (p, a))

        plt.imshow(xTest[i])
        plt.show()


# Active the tensorboard for callbacks
def activateTensorBoard():
    NAME = 'flowers-cnn-64x2-{}'.format(int(time.time()))

    tensorBoard = TensorBoard(log_dir='logs/{}'.format(NAME))

    return tensorBoard


# Save the model once trained
def saveModel(model):
    model.save('FlowerClassifierModel')


if __name__ == '__main__':
    X, y = getData()
    model = createNeuralNet(X)
    tensorBoard = activateTensorBoard()
    trainModel(model, X, y, tensorBoard)

    testModel(model, X, y)

    if input('\nSave? [y/n]\n').lower() == 'y':
        saveModel(model)
