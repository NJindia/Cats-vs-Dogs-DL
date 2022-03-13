#!/usr/bin/env python
# coding: utf-8
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from keras import models, layers, Sequential
from os.path import join, isfile
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import KFold
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import datetime

# def getImgs(parent_folder_path):
#     imgs = []
#     for f in listdir(parent_folder_path):
#         if isfile(join(parent_folder_path, f)):
#             path = join(parent_folder_path, f)
#             img = load_img(path, target_size=(200, 200))
#             img_arr = img_to_array(img)
#         imgs.append(img_arr)
#     return np.array(imgs)
#
#
# cat_imgs = getImgs('/home/techn/Pictures/PetImages/Cat')[:5000]
# dog_imgs = getImgs('/home/techn/Pictures/PetImages/Dog')[:5000]
# # cat_imgs = getImgs('D:\\Downloads\\kagglecatsanddogs_3367a\\PetImages\\Cat')[:5000]
# # dog_imgs = getImgs('D:\\Downloads\\kagglecatsanddogs_3367a\\PetImages\\Dog')[:5000]
# np.save('cat_photos.npy', cat_imgs)
# np.save('dog_photos.npy', dog_imgs)
cat_imgs = np.load('cat_photos.npy')
dog_imgs = np.load('dog_photos.npy')
len(cat_imgs), len(dog_imgs)

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.tight_layout()
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.legend()
    plt.show()


def kfold(network, cat_imgs, dog_imgs, batch_size):
    kf = KFold(n_splits=5)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    n = 0
    fold_times = []
    histories = []
    for train_idx, test_idx in kf.split(cat_imgs):
        start = datetime.datetime.now()
        print(f"Fold {n}")
        n += 1
        print(f"Test Index Start:{test_idx[0]}")
        print(f"Test Set Size:{len(test_idx)}")
        train_imgs = np.append(cat_imgs[train_idx], dog_imgs[train_idx], axis=0)
        train_labels = np.append(np.full(len(train_idx), 0), np.full(len(train_idx), 1))

        test_imgs = np.append(cat_imgs[test_idx], dog_imgs[test_idx], axis=0)
        test_labels = np.append(np.full(len(test_idx), 0), np.full(len(test_idx), 1))

        train_it = datagen.flow(train_imgs, train_labels, batch_size=batch_size)
        test_it = datagen.flow(test_imgs, test_labels, batch_size=batch_size)
        history = network.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it,
                              validation_steps=len(test_it), epochs=20, verbose=0)
        histories.append(history)
        train_loss, train_acc = network.evaluate(train_imgs, train_labels, verbose=0)
        test_loss, test_acc = network.evaluate(test_imgs, test_labels, verbose=0)
        print(f"Train Loss: {train_loss}; Train Accuracy: {train_acc}")
        print(f"Test Loss: {test_loss}; Test Accuracy: {test_acc}")
        fold_time = datetime.datetime.now() - start
        fold_times.append(fold_time.total_seconds())
        print("fold time: ", fold_time)
    plt.tight_layout()
    for n in range(len(histories)):
        plt.subplot(5, 2, 1 + 2 * n)
        plt.title(f'Fold {n} Cross Entropy Loss')
        plt.plot(histories[n].history['loss'], color='blue', label='train')
        plt.plot(histories[n].history['val_loss'], color='orange', label='test')

        plt.subplot(5, 2, (2 * n) + 2)
        plt.title(f'Fold {n} Classification Accuracy')
        plt.plot(histories[n].history['accuracy'], color='blue', label='train')
        plt.plot(histories[n].history['val_accuracy'], color='orange', label='test')
    plt.legend()
    plt.show()

    print(datetime.timedelta(seconds=np.average(fold_times)))


def build_network():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                       input_shape=(200, 200, 3)))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Flatten())
    network.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    network.add(Dense(1, activation='sigmoid'))
    return network


# In[4]:


opt = SGD(learning_rate=0.001, momentum=0.9)
network = build_network()
network.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

for lr in [0.01, 0.05]:
    print("Learning Rate: ", lr)
    opt = SGD(learning_rate=lr, momentum=0.9)
    network = build_network()
    network.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    kfold(network, cat_imgs, dog_imgs, 32)
