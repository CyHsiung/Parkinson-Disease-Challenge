import numpy as np
from os.path import basename, splitext, join
import json
import math
import os

from getData_xyz import read_train_file, read_test_file
from test import test_model, test_model_prior, ROC_score

#keras import
import keras

from keras.models import model_from_json
from keras.layers.wrappers import TimeDistributed
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models_xyz")
feats_dir = join(project_dir, "feats")

trainRatio = 1
trainFileName = 'label_train_data_train.csv'
testFileName = 'label_train_data_valid.csv'
freq = 80
secLength = 43


loadWeight = False

loadModelName = 'CNN_dropout_xyz_2'
saveModelName = '1234_sub2'

models_dir = join(models_dir, loadModelName)
# parameter
nEpoch = 400
fileName = 'label_data.csv'

model_structure = join(models_dir, 'model_structure')

model_weight_path = join(models_dir, 'initialize_weight')


# Read file 
X, Y = read_train_file(trainFileName, freq, secLength)

trainLength = int(trainRatio * len(X))
X_train = X
Y_train = Y

testList = read_test_file(testFileName, freq, secLength)
# new model directory for new parameter
models_dir = join(project_dir, "models_xyz")

# make model directory
os.makedirs(join(models_dir, saveModelName))
models_dir = join(models_dir, saveModelName)



# load model structure and weight
with open(model_structure) as json_file:
    model_architecture = json.load(json_file)

model = model_from_json(model_architecture)

if loadWeight:
    model.load_weights(model_weight_path, by_name = True)

model.compile(loss='mean_squared_error', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])

# model.get_weights()
model.summary()

# Save model structure
model_architecture = model.to_json()

with open(models_dir+'/model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)

# acc_highest = prior[0]
acc_highest = 0
print(acc_highest)
for i in range(nEpoch):
    model.fit([X_train[0], X_train[1], X_train[2]], Y_train,epochs = 1) 
    # test
    # acc = test_model(model, testList, 0.7)
    acc = ROC_score(model, testList)
    print(acc)
    if (acc > acc_highest):
        # model.save_weights(join(models_dir,'Epoch_'+str(i+1)+'_ACC_'+str(acc)))
        model.save_weights(join(models_dir,'Epoch_'+str(i+1) + '_acc_' + str(acc)))
        acc_highest = acc
