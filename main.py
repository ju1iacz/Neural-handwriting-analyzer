import keras.callbacks
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

count = 0
images = []
classNumber = []
myList = os.listdir('data')
print("Detected classes:", len(myList))                                     #number of detected classes
number = len(myList)                                                        #number of folders in 'data'
print("Importing classes ...")

for x in range(0, number):                                                  #x - folder name
    picList = os.listdir('data' + "/" + str(x))
    for y in picList:                                                       #y - image name
        current = cv2.imread('data' + "/" + str(x) + "/" + y)
        current = cv2.resize(current, (32, 32))                             #reducing the image size
        images.append(current)
        classNumber.append(count)
    print(count, end = " ")
    count += 1

print(" ")
print("Total number of images:", len(images))

images = np.array(images)
classNumber = np.array(classNumber)
#print(images.shape)
#print(classNumber.shape)

#Spliting the data
x_train, x_test, y_train, y_test = train_test_split(images, classNumber, test_size = 0.2) #80% training, 20% testing
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

#print(x_train.shape)
#print(x_test.shape)
#print(x_val.shape)

#plt.figure(figsize = (10, 5)) #wykres
#plt.bar(range(0, number), samples)
#plt.title("Number of images for each class")
#plt.xlabel("Class ID")
#plt.ylabel("Number of images")
#plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #shades of grey
    img = cv2.equalizeHist(img)                 #histogram equalization
    img = img/255                               #black and white
    return img

#img = preProcessing(X_train[10])
#img = cv2.resize(img, (300, 300))
#cv2.imshow("PreProcessed", img)
#cv2.waitKey(0)

x_train = np.array(list(map(preProcessing, x_train)))
x_test = np.array(list(map(preProcessing, x_test)))
x_val = np.array(list(map(preProcessing, x_val)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) #adding depth
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.2,
                             rotation_range = 10)
dataGen.fit(x_train)

y_train = to_categorical(y_train, number)
y_test = to_categorical(y_test, number)
y_val = to_categorical(y_val, number)

def myModel():
    filters = 60
    size1 = (5, 5)
    size2 = (3, 3)
    pool = (2, 2)
    node = 500

    model = Sequential()
    model.add((Conv2D(filters, size1, input_shape = (32, 32, 1), activation = 'relu')))

    model.add(MaxPooling2D(pool_size = pool))
    model.add((Conv2D(filters//2, size2, activation='relu')))
    model.add(MaxPooling2D(pool_size = pool))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(node, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(number, activation = 'softmax'))
    model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(x_train, y_train,                    #network training
                                           batch_size = 6),
                                           steps_per_epoch = 90,
                                           epochs = 25,
                                           validation_data = (x_val, y_val),
                                           shuffle = 1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
#plt.show()

score = model.evaluate(x_test, y_test, verbose = 0)
print('Test Score = ', score[0])
print('Test Accyracy = ', score[1])

model.save('model.h5')
print("Model successfully saved")