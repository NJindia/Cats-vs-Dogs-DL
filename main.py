import sys

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from keras import models, layers, Sequential
from os.path import join, isfile

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator


def define_network():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                       input_shape=(200, 200, 3)))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    network.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return network


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


def getImgs(parent_folder_path):
    imgs = []
    for f in listdir(parent_folder_path):
        if isfile(join(parent_folder_path, f)):
            path = join(parent_folder_path, f)
            img = load_img(path, target_size=(200, 200))
            img_arr = img_to_array(img)
        imgs.append(img_arr)
    return np.array(imgs)


def kfold(network, cat_imgs, dog_imgs):
    kf = KFold(n_splits=5)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    for train_idx, test_idx in kf.split(cat_imgs):
        train_imgs = np.append(cat_imgs[train_idx], dog_imgs[train_idx], axis=0)
        train_labels = np.append(np.full(len(train_idx), 0), np.full(len(train_idx), 1))

        test_imgs = np.append(cat_imgs[test_idx], dog_imgs[test_idx], axis=0)
        test_labels = np.append(np.full(len(test_idx), 0), np.full(len(test_idx), 1))

        train_it = datagen.flow(train_imgs, train_labels, batch_size=64)
        test_it = datagen.flow(test_imgs, test_labels, batch_size=64)
        history = network.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it,
                              validation_steps=len(test_it), epochs=20, verbose=3)

        train_loss, train_acc = network.evaluate(train_imgs, train_labels)
        test_loss, test_acc = network.evaluate(test_imgs, test_labels)


if __name__ == '__main__':
    # Import Data
    cat_imgs = getImgs('/home/techn/Pictures/PetImages/Cat')
    dog_imgs = getImgs('/home/techn/Pictures/PetImages/Dog')
    np.save('cat_photos.npy', cat_imgs)
    np.save('dog_photos.npy', dog_imgs)

    cat_imgs = np.load('cat_photos.npy')
    dog_imgs = np.load('dog_photos.npy')

    network = define_network()
    # fit model
    kfold(network, cat_imgs, dog_imgs)
