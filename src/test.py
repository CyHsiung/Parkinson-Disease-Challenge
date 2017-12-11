import numpy as np
from os.path import basename, splitext, join
import json
import math
import os
from sklearn.metrics import roc_auc_score as roc
from scipy.stats import mode
from getData_xyz import read_test_file, read_train_file

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


def weak_evaluate(y_predict):
    for k in range(y_predict.shape[0]):
        if(y_predict[k][0] > y_predict[k][1]):
            y_predict[k][0] = 1
            y_predict[k][1] = 0
        else :
            y_predict[k][0] = 0
            y_predict[k][1] = 1
    return y_predict
# larger bias -> easier to predict class 0
def bias_evaluate(y_predict, bias):
    for k in range(y_predict.shape[0]):
        if(y_predict[k][0] > 1 - bias):
            y_predict[k][0] = 1 
            y_predict[k][1] = 0
        else :
            y_predict[k][0] = 0
            y_predict[k][1] = 1
    return y_predict
    

def strong_evaluate(y_predict):
    for i in range(y_predict.shape[0]):
        for j in range(y_predict.shape[1]):
            if(y_predict[i][j] > 0.7):
                y_predict[i][j] = 1
            if(y_predict[i][j] < 0.3):
                y_predict[i][j] = 0
    return y_predict

def test_model(model, testList, bias):
    acc = 0
    totalNum = len(testList)    
    error = [0 ,0]

    for testPair in testList:
        y_predictList = model.predict([np.asarray(testPair[0][0]), np.asarray(testPair[0][1]), np.asarray(testPair[0][2])])
        y_predictList = bias_evaluate(y_predictList, bias)
        predict = 0
        predZeroNum = 0
        predOneNum = 0
        
        for y_predict in y_predictList:
            # calculate accuracy
            if y_predict[0] == 1:
                predZeroNum += 1
                
            else:
                predOneNum += 1
        if predOneNum > predZeroNum:
            predict = 1
        if(predict == testPair[1]):
            acc += 1
        else:
            error[int(testPair[1])] += 1
    print(error)
    return round(float(acc)/totalNum, 5)

def test_model_prior(model, testList, prior):
    acc = 0
    totalNum = len(testList)    
    error = [0 ,0]

    for testPair in testList:
        # likeliRatio > 1: choose 1; < 1 : choose 0
        likeliRatio = float(prior[1]/prior[0])
        y_predict = model.predict([np.asarray(testPair[0][0]), np.asarray(testPair[0][1]), np.asarray(testPair[0][2])])
        for predElement in y_predict:
            likeliRatio *= (predElement[1]/predElement[0])
        if likeliRatio > 1:
            predict = 1
        else:
            predict = 0 
        if(predict == testPair[1]):
            acc += 1
        else:
            error[int(testPair[1])] += 1
    print(error)
    return round(float(acc)/totalNum, 5)
            
def ROC_score(model, testList):
    
    y_true = []
    y_score = []
    for testPair in testList:
        # likeliRatio > 1: choose 1; < 1 : choose 0
        if(testPair[1] == 0):
            y_true.append([1, 0])
        else:
            y_true.append([0, 1])
        y_predict = model.predict([np.asarray(testPair[0][0]), np.asarray(testPair[0][1]), np.asarray(testPair[0][2])])
        y_score.append(np.median(y_predict, axis = 0))
        # transform to numpy
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)

    # print(y_score)
    x = round(roc(y_true, y_score), 5)
    print('roc: ', x)
    return x
def select_threshold_by_ROC(model, testList):
    biasList = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    rBias = 0
    rocMax = 0
    for bias in biasList:
        y_true = []
        for testPair in testList:
            y_predictList = model.predict([np.asarray(testPair[0][0]), np.asarray(testPair[0][1]), np.asarray(testPair[0][2])])
            y_predictList = bias_evaluate(y_predictList, bias)
            mod = mode(y_predictList)
            testPair[1] = mod[0][0][1]
        curRoc = ROC_score(model, testList)
        if curRoc > rocMax:
            rocMax = curRoc
            rBias = bias
    print('bias: ', rBias)
    return rBias

def count_prior(testList):
    totalNum = len(testList)
    countZero = 0
    for pair in testList:
        if pair[1] == 0:
            countZero += 1
    prior = float(countZero/totalNum)
    return [prior, 1- prior]
            


if __name__ == '__main__':
    testFileName = 'label_test_data.csv'
    trainFileName = 'label_train_data.csv'
    # modelName ='M2_both_roc_0.927'
    # weightName = 'Epoch_41_acc_0.92797'
    modelName ='1234_sub2'
    weightName = 'Epoch_8_acc_0.94088'

    models_dir = join(models_dir, modelName)
    # parameter

    # load variable (change with different model)
    model_structure = join(models_dir, 'model_structure')

    model_weight_path = join(models_dir, weightName)

    # load model structure and weight
    with open(model_structure) as json_file:
        model_architecture = json.load(json_file)

    model = model_from_json(model_architecture)

    model.load_weights(model_weight_path, by_name=False)
    # load test file
    testList = read_test_file(testFileName, 80, 43)
    
    # count percentage of 0 
    prior = count_prior(testList)   
    ROC_score(model, testList)
    # bias = select_threshold_by_ROC(model, testList)
    
    print('blind guess 0: ', prior[0])
    testList = read_test_file(testFileName, 80, 43)
    x= test_model(model, testList, 0.6)
    print('accuracy:', x)
    x= test_model(model, testList, 0.7)
    print('accuracy:', x)
    x= test_model(model, testList, 0.8)
    print('accuracy:', x)
    x= test_model(model, testList, 0.9)
    print('accuracy:', x)
           
    
