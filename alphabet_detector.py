
'''
#____________Sign Language Alphabet Recognition_____________


#Model making imports
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import keras

#Model loading imports
import json
import numpy as np
from keras import models
from keras.models import model_from_json, load_model

#Data procesing imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pickle

#Visualization imports
import os
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from IPython.display import display
#%matplotlib inline


#_____ SETTING THE DATA PATH ______
DATADIR = "/Users/ewa_anna_szyszka/Desktop/Code/ImageRecognition/data"

#_____ CATEGORIES OF CLASSES ______
CATEGORIES = ["A", "B", "C", "D", "E","F", "G", "H", "I", "J","K", "L", "M",'N','O','P','R','S','T','U','W','X','Y','Z']

#_____ SETTING UP THE TRAINING DATA ______

#setting the size of the images to 50x50
IMG_SIZE = 50

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        #One hot encoding
        one_hot_target = np.zeros(len(CATEGORIES))
        class_num = CATEGORIES.index(category)
        one_hot_target[class_num] = 1

        for img in os.listdir(path):
            try:
                #resizing the images and attaching one hot encoded values
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,one_hot_target])
            except Exception as e:
                pass

create_training_data()

#___ SHUFFLING THE DATA TO IMPROVE THE TRAINING QUALITY _____
random.shuffle(training_data)

#____PRINTING SAMPLE DATA_____
for sample in training_data[:1]:
    print("This is one hot encoded label: \n", sample[1])
    print("This is np.array of an image: \n", sample[0])

X = [] #feature set
y = [] #labels

for features, label in training_data:
    X.append(features)
    y.append(np.asarray(label)) #converting y to np array

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE, 3) # 1 because it is a gray scale




#____ NORMALIZING THE IMAGE DATA _____
X = X/255.0

#_____ PRINTING SHAPE OF X AND Y _____
X = np.asarray(X)
y = np.asarray(y)
print(X.shape,y.shape)

#____ TEST-TRAIN SPLIT THE DATA _____

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#____ LENGTH OF THE TEST-TRAIN DATA_____

print("\n X train:",len(X_train),"\n y train:",len(y_train),"\n X test:" ,len(X_test),"\n y test:",len(y_test))

#_____RESHAPING THE DATA ______

X_train = X_train.reshape(45985,50,50,3)
X_test = X_test.reshape(22650,50,50,3)

#____ LENGTH OF THE TEST-TRAIN DATA AFTER RESHAPING____

print("\n X train:",len(X_train),"\n y train:",len(y_train),"\n X test:" ,len(X_test),"\n y test:",len(y_test))

#_____ CONVERTING TO NP.ARRAY _____
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
type(y_train)

#________ SETTING UP THE SIMPLIFIED MODEL _______
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(50,50,3)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(24, activation="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)


#How to assess the model:
#Model loss and model accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



#________ MAKING A TEST PREDICTION _______

first_predictions = model.predict(X_test[:4])

#Showing one hot encoded label that the prediction was made on
print(y_test[0])

"""Showing the image that the prediction was made on"""
plt.imshow(Image.fromarray(X_test[0],'RGB'), interpolation='nearest')
plt.axis("off")
plt.show()

#Showing was was actually predicted
print(first_predictions[0])


#____ SAVINGS THE TRAINED MODEL ____
model.save('my_model.h5')

model_json = model.to_json()
with open("model.json", "w") as json_file:
     json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

'''


#What is above needs to be executed in google colab --> the training process of the model which is then saved to model.h5

#Next you need to download the saved model.h5 and model.json from google drive where the model was saved and place it in Desktop

#Then, load the model and do the predictions using the camera

from keras import models
from keras.models import load_model
from keras.models import model_from_json, load_model
import glob


#_______ LOADING THE TRAINED MODEL TO MAKE PREDICTIONS ON UNSEEN DATA ____
model = models.load_model('my_model.h5')


#______ OBJECT OPENING TRAINED MODEL AND PREDICTING IMAGES FROM CAPTURED DATA_____

class SignLanguageModel(object):

    LETTER_LIST = ["A", "B", "C", "D", "E","F" ]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_letter(self, img):
        self.preds = self.loaded_model.predict(img)
        print (self.preds)
        return SignLanguageModel.LETTER_LIST[np.argmax(self.preds)]

#______ TIME TO CAPTURE DATA _____

import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)


    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed  ---> somethig is off here as the images are not being saves to the correct location
        img_name = "Users/ewa_anna_szyszka/Desktop/Code/ImageRecognition/capture_for_recognition/opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

'''

#Creating the images for recognition

images=glob.glob("Desktop/Code/ImageRecognition/datacapture/*.png")

images_for_recognition = []

for image in images:
    img = Image.open(image)
    images_for_recognition.append(img)
    display(img)


#___MAKING THE PREDICTION ON THE CAPTURED DATA _____
model_json_file =  "/Users/ewa_anna_szyszka/Desktop/model.json"
model_weights_file = "/Users/ewa_anna_szyszka/Desktop/my_model.h5"

#Setting up the
for i in images_for_recognition:
    new_array = cv2.resize(np.array(i), (50, 50))
    new_array = new_array.reshape(1,50,50,3)
    a = SignLanguageModel(model_json_file, model_weights_file)
    print(a.predict_letter(new_array))
'''
