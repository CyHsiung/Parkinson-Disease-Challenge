'''
Create ur own model and then decide model name for its directory name(parameter modelName below)
model_structure for training model
and model_structure_feature for feature printing and make prediction

'''
import glob
import os
import numpy as np
from os.path import basename, splitext, join, dirname
import json
import math
# np.random.seed(123)  # for reproducibility

from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from keras.layers import add
from getData_xyz import read_train_file

fileName = 'label_data.csv'
modelName ='CNN_dropout_xyz_123'
freq = 80
secLength = 43

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models_xyz")
feats_dir = join(project_dir, "feats")


# make model directory
os.makedirs(join(models_dir, modelName))

models_dir = join(models_dir, modelName)

# Parameter
timeLength = secLength

num_class = 2

# Model
# x_batch, y_batch, prior = read_train_file(fileName, freq, secLength)
# print(x_batch.shape)
# n, channel, freqBand, timeLength = x_batch.shape
freqBand = math.floor(freq/2) + 1
channel = 1

# x part
inp_x = Input(shape = (channel, freqBand, timeLength)) # depth goes last in TensorFlow back-end (first in Theano)

conv_1_x = Conv2D(filters = 16, kernel_size = (6, 8), strides = (2, 2), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_1_x')(inp_x)
pool_1_x = MaxPooling2D(pool_size=(1, 1), data_format='channels_first')(conv_1_x)
drop_1_x = Dropout(rate = 0.3)(pool_1_x)

conv_2_x = Conv2D(filters = 32, kernel_size = (4, 6), strides = (1, 1), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_2_x')(drop_1_x)
pool_2_x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv_2_x)
drop_2_x = Dropout(rate = 0.3)(pool_2_x)

# y part
inp_y = Input(shape = (channel, freqBand, timeLength)) # depth goes last in TensorFlow back-end (first in Theano)

conv_1_y = Conv2D(filters = 16, kernel_size = (6, 8), strides = (2, 2), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_1_y')(inp_y)
pool_1_y = MaxPooling2D(pool_size=(1, 1), data_format='channels_first')(conv_1_y)
drop_1_y = Dropout(rate = 0.3)(pool_1_y)

conv_2_y = Conv2D(filters = 32, kernel_size = (4, 6), strides = (1, 1), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_2_y')(drop_1_y)
pool_2_y = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv_2_y)
drop_2_y = Dropout(rate = 0.3)(pool_2_y)

# z part
inp_z = Input(shape = (channel, freqBand, timeLength)) # depth goes last in TensorFlow back-end (first in Theano)

conv_1_z = Conv2D(filters = 16, kernel_size = (6, 8), strides = (2, 2), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_1_z')(inp_z)
pool_1_z = MaxPooling2D(pool_size=(1, 1), data_format='channels_first')(conv_1_z)
drop_1_z = Dropout(rate = 0.3)(pool_1_z)

conv_2_z = Conv2D(filters = 32, kernel_size = (4, 6), strides = (1, 1), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_2_z')(drop_1_z)
pool_2_z = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv_2_z)
drop_2_z = Dropout(rate = 0.3)(pool_2_z)

merged = Concatenate(axis = 1)([drop_2_x, drop_2_y, drop_2_z])

conv_3 = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_3')(merged)
pool_3 = MaxPooling2D(pool_size=(1, 1), data_format='channels_first')(conv_3)
drop_3= Dropout(rate = 0.3)(pool_3)

conv_4 = Conv2D(filters = 32, kernel_size = (2, 2), strides = (1, 1), padding = 'valid', data_format = "channels_first", activation='relu', name = 'conv_4')(drop_3)
pool_4 = MaxPooling2D(pool_size=(1, 1), data_format='channels_first')(conv_4)
drop_4= Dropout(rate = 0.3)(pool_4)
flat = Flatten()(drop_4)
hidden = Dense(units = 64, activation='relu', name = 'hid_1')(flat)
# drop_3 = Dropout(rate = 0.2)
out = Dense(units = num_class, activation='softmax', name = 'output')(hidden)

model = Model(inputs = [inp_x, inp_y, inp_z], outputs = out)

model.summary()

model_feature = Model(inputs = [inp_x, inp_y, inp_z], outputs = [out])

# current stop 
input('qq')

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])

# Save model structure
model_architecture = model.to_json()

with open(models_dir+'/model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)

# model to print feature
model_architecture_feature = model_feature.to_json()

with open(models_dir+'/model_structure_feature', 'w') as outfile:
    json.dump(model_architecture_feature, outfile)


text_file = open(join(models_dir, "parameter"), "w")
text_file.write("secLength: %s\n" % (str(secLength)))
text_file.write("freq: %s\n" % (str(freq)))
text_file.close()
