#!/usr/bin/env python
# coding: utf-8

#load libraries

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob


img_rows = 224
img_cols = 224 



train_path = 'mlops/dataset/train'
validation_path = 'mlops/dataset/test'



model = VGG16(input_shape = (img_rows,img_cols,3), weights='imagenet', include_top=False)


for layer in model.layers:
    layer.trainable = False


folders = glob('mlops/dataset/train/*')


#using image augmentation 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=20,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)



#change batchsize 
batch_size = 8
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory(validation_path,
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')
num_pixels = (224, 224)
num_pixels



#function for adding top layers

def add_model(neurons,num_classes):
    top_model = Flatten()(model.output)
    top_model = Dense(units=neurons,input_dim=num_pixels,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    new_model = Model(inputs=model.input, outputs=top_model)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.summary()
    return new_model


#call add_model function
#If you have enough resources you can start with 128 neurons

neurons = 5
num_classes = len(folders)
model = add_model(neurons,num_classes)
accuracy = 0.0


#function for accuracy of model

def build_model(epoch,spepoch,val_steps):
    print("Training model...")
    history = model.fit_generator(training_set,validation_data=validation_set,
    epochs=epoch,steps_per_epoch=spepoch,validation_steps=val_steps)
    
    test_accuracy=history.history['val_acc'][-1]
    print(test_accuracy)
    accuracy=test_accuracy*100
    print(accuracy)
    return accuracy

#call accuracy function

epoch = 3
spepoch = len(training_set)
val_steps = len(validation_set)

accuracy = build_model(epoch,spepoch,val_steps)
count = 0
best_accuracy = accuracy
best_neurons = neurons



#function for resetting weights
def resetWeights():
    print("Resetting all the weights.....")
    w = model.get_weights()
    w = [[j*0 for j in i] for i in w]
    model.set_weights(w)



#if accuracy level not satisfied change hyper parameters
#maximum try = 5

while accuracy < 85 and count < 5:
    print("Updating model....")
    model = add_model(neurons*2,num_classes)
    neurons = neurons*2
    count = count + 1
    epoch = epoch + 2
    accuracy = build_model(epoch,spepoch,val_steps)
    if best_accuracy < accuracy:
        best_accuracy = accuracy
        best_neurons = neurons
    print()
    resetWeights()


#save best model


print("******************************")
print("Best neurons:")
print(best_neurons)
print("Best Accuracy:")
print(best_accuracy)
if count > 1:
	model = add_model(best_neurons,num_classes)
	build_model(epoch,spepoch,val_steps)
model.save("updated_model.h5")
print("Model Saved!")




# store accuracy result
print("hey")
print(best_accuracy)
file1=open("result.txt","w")
print("hey")
file1=write(str(best_neurons,best_accuracy))
print("hey")
file1.close()

