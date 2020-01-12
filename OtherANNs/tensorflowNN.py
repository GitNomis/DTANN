# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 16:55:58 2020

@author: pille

Doc:
    https://www.tensorflow.org/api_docs/python/tf/keras/Sequential



"""
import tensorflow as tf
import keras

NN = tf.keras.Sequential()

NN.add(keras.layers.Dense(16, activation='relu'))
NN.add(keras.layers.Dense(20))
#print(NN.layers)