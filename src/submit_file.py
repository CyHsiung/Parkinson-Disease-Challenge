import numpy as np
import pandas as pd

from os.path import basename, splitext, join
import json
import math
import os
from sklearn.metrics import roc_auc_score as roc
from scipy.stats import mode
from getData_xyz import read_test_file, read_X_single_file

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


def test_model(model, testList, bias):
    acc = 0
    totalNum = len(testList)    
    error = [0 ,0]
def load_model(models_dir, modelName, weightName):

    models_dir = join(models_dir, modelName)
    # load variable (change with different model)
    model_structure = join(models_dir, 'model_structure')

    model_weight_path = join(models_dir, weightName)

    # load model structure and weight
    with open(model_structure) as json_file:
        model_architecture = json.load(json_file)

    model = model_from_json(model_architecture)

    model.load_weights(model_weight_path, by_name=False)
    return model

def fill_nan(idFeaList, aveFea):
    for pair in idFeaList:
        if len(pair) == 1:
            pair.append(aveFea)
    return idFeaList

def get_feature_id_pair( file_df, freq, secLength, model, feature, feaLength):
    # testFile_df = file_df.ix[(file_df['device'] == 'GENEActiv') & ((file_df['bradykinesiaScore'] == 1) | (file_df['bradykinesiaScore'] == 0))]
    testFile_df = file_df.ix[(file_df[feature] == 1) | (file_df[feature] == 0) | (file_df[feature] == -1)]
    r, c = testFile_df.shape
    idFeaList = []
    # To calculate average
    totalSum = np.zeros((feaLength))
    totalNum = 0
    for i in range(r):
        pair = []
        pair.append(testFile_df.iloc[i]['dataFileHandleId'])
        X = read_X_single_file(i, testFile_df, freq, secLength)
        if not X: 
            idFeaList.append(pair)
            continue
        x_batch = X
        y_predict = model.predict([x_batch[0], x_batch[1], x_batch[2]]) 
        y_predict = np.median(y_predict, axis = 0)
        pair.append(y_predict)
        # calculate feature average to fill NAN
        totalSum += y_predict
        totalNum += 1
        
        idFeaList.append(pair)
        fill_nan(idFeaList, totalSum/totalNum)

    print(len(idFeaList))
    return idFeaList
def transform_list_to_dataFrame(idFeaList):
    dataFrame  = pd.DataFrame([])
    for pair in idFeaList:
        element = {'dataFileHandleId' : pair[0]}
        for i in range(len(pair[1])):
            element['feat' + str(i + 1)] = pair[1][i]
        dataFrame = dataFrame.append(pd.DataFrame(element, index=[0]), ignore_index=True)
    return dataFrame
    
        


if __name__ == '__main__':
    fileName = ['label_data.csv', 'test_data_bradykinesiaScore.csv']
    modelName ='M2_both_roc_0.927'
    weightName = 'Epoch_41_acc_0.92797'
    targetClass = 'dataFileHandleId'   
    feature = 'bradykinesiaScore' 
    resultName = modelName + '.csv'   
    referenceFile = 'bradykinesiaSubmissionTemplate.csv'

    # unchange parameter
    freq = 80
    secLength = 43
 

    model = load_model(models_dir, modelName, weightName)
    dataFrame = pd.DataFrame([])
    for name in fileName:
        file_df = pd.read_csv(join(corpus_dir, name))
        idFeaList = get_feature_id_pair(file_df, freq, secLength, model, feature, 2)
        dataFrame = dataFrame.append(transform_list_to_dataFrame(idFeaList), ignore_index = True)
        print(dataFrame.shape)
    # to dictionary
    dic = {}
    r, c = dataFrame.shape
    for i in range(r):
        dic[dataFrame.iloc[i]['dataFileHandleId']] = dataFrame.iloc[i][:]
    # Use id in reference to build the table
    submit_df = pd.DataFrame([])
    reference_df = pd.read_csv(join(corpus_dir, referenceFile))
    print('reference length: ', reference_df.shape[0])
    for i in range(reference_df.shape[0]):
        if reference_df.iloc[i]['dataFileHandleId'] not in dic:
            print('poor key: ', reference_df.iloc[i]['dataFileHandlwId'])
            continue
        submit_df = submit_df.append(dic[reference_df.iloc[i]['dataFileHandleId']])
    

    submit_df.to_csv(join(result_dir, resultName), index=False)
