import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from keras import models, layers, Sequential
from os.path import join, isfile

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import KFold
from keras.preprocessing.image import load_img, img_to_array

def getImgs(parent_folder_path):
    imgs = []
    labels = []
    for f in listdir(parent_folder_path):
        if isfile(join(parent_folder_path, f)):
            path = join(parent_folder_path, f)
            img = load_img(path, target_size=(200,200), color_mode="grayscale")
            img_arr = img_to_array(img)
            if 'cat' in f: labels.append(0)
            else: labels.append(1)
        imgs.append(img_arr)
    return np.array(imgs), np.array(labels)

def kfold(network, imgs, labels):
    kf = KFold(n_splits=5)
    for train_idx, test_idx in kf.split(imgs):
        train_imgs = imgs[train_idx]
        train_labels = labels[train_idx]
        test_imgs = imgs[test_idx]
        test_labels = labels[test_idx]
        network.fit(train_imgs, train_labels, epochs=5, batch_size=128)
        train_loss, train_acc = network.evaluate(train_imgs, train_labels)
        test_loss, test_acc = network.evaluate(test_imgs, test_labels)
        ...

if __name__ == '__main__':
    # Import Data
    imgs, labels = getImgs('D:\\Documents\\dogs-vs-cats\\train')
    np.save('dogs_vs_cats_photos.npy', imgs)
    np.save('dogs_vs_cats_labels.npy', labels)

    imgs = np.load('dogs_vs_cats_photos.npy')
    labels = np.load('dogs_vs_cats_labels.npy')

    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 1)))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(1, activation='sigmoid'))
    network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    kfold(network, imgs, labels)