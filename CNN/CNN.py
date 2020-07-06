"""Created on Tue May 12 11:22:09 2020"""

## ------------Part1 - Building the CNN ----------------------------------------------

'''importing the libraries and packages'''
from keras.models import Sequential
from keras.layers import Convolution2D    #for addding convolution layer(time is the 3rd dimension in videos)
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

'''initializing the cnn''' #same as ann
classifier = Sequential()  #cnn initialized

''' step1- convolution'''
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))
# Conv2D(32, (3, 3), input_shape=(64, 64, 3..., activation="relu")
# input_shape=(3,64,64) if backend==theano 

''' step2 - pooling'''
classifier.add(MaxPooling2D(pool_size=(2,2)))
# reducing size of feature maps (in maxpooling we take max of value caught in  k*k matrix pool)


'''label: ADDMORE'''
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


''' step3 - flatenning'''
classifier.add(Flatten())

''' step4 - full connection'''
classifier.add(Dense(output_dim = 128, activation='relu'))      #fullyConnectedLayer (hiddenLayer)
classifier.add(Dense(output_dim = 1, activation='sigmoid'))     # output layer
#Dense(activation="relu", units=128)

'''compiling the CNN'''
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam = stochastic gradient
#binary_crossentroy because output is binary and also logarithm fuction is used.
#metrics is performance metrics


## ------------Part2 - fitting the CNN to images ----------------------------------------------

# image augmetation: amout of training images is augmented (copied, flipped, sheared etc)
# enriches the result without having more number of images\
# feature scaling is an essential part of deep learning and computer vision

from keras.preprocessing.image import ImageDataGenerator

# object that will be used to augment the training set images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# object that will be used to augment the test set images
test_datagen = ImageDataGenerator(rescale=1./255)

# apply augmentation on training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),  #size of images expected in cnn model
                                                batch_size=32,         
                                                class_mode='binary')

# apply augmentation on test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                steps_per_epoch=8000,   #no of images in training set
                epochs=25,              #no of epochs
                validation_data=test_set,    # on which you want to evaluate the performance of test set
                validation_steps=2000)       #no of images in test set  


'''now in order to improve our model, we can add either more convolution layers
or fully connected layer. Better we add more convolution layers. so goto ADDMORE'''

# more pixels(bigger than 64*64) = more information = more accuracy
