import pandas as pd
import numpy as np
from os.path import basename, splitext, join
import re
import librosa
import math

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
feats_dir = join(project_dir, "feats")

sampleNormalize = True
inputNormalize = False

catagoryColName = 'dyskinesiaScore'

def read_train_file(fileName, freq, secLength):
    x_train = [[], [], []]
    y_train = []
    x_zero = read_X_file_train(fileName, 0, freq, secLength)
    x_one = read_X_file_train(fileName, 1, freq, secLength)
    appendNum = max(x_zero[0].shape[0], x_one[0].shape[0])
    print(len(x_zero))
    print(x_zero[0].shape)
    
    for i in range(appendNum):
        '''
        for j in range(ratio):
            x_train.append(x_zero[ratio * i + j, :, :])
            y_train.append([1, 0])
        '''
        x_train[0].append(x_zero[0][i % x_zero[0].shape[0], :, :, :])
        x_train[1].append(x_zero[1][i % x_zero[0].shape[0], :, :, :])
        x_train[2].append(x_zero[2][i % x_zero[0].shape[0], :, :, :])
        y_train.append([1, 0])
        x_train[0].append(x_one[0][i % x_one[0].shape[0], :, :, :])
        x_train[1].append(x_one[1][i % x_one[0].shape[0], :, :, :])
        x_train[2].append(x_one[2][i % x_one[0].shape[0], :, :, :])
        y_train.append([0, 1])
        
        x_batch = np.asarray(x_train)
        y_batch = np.asarray(y_train)

    # prior = count_prior(fileName)
    return x_batch, y_batch

def count_prior(fileName):
    file_df = pd.read_csv(join(corpus_dir, fileName))
    # zero_df = file_df.ix[(file_df['device'] == 'GENEActiv') & (file_df[catagoryColName] == 0)]
    # one_df = file_df.ix[(file_df['device'] == 'GENEActiv') & (file_df[catagoryColName] == 1)]
    zero_df = file_df.ix[file_df[catagoryColName] == 0]
    one_df = file_df.ix[file_df[catagoryColName] == 1]
    totalNum = one_df.shape[0] + zero_df.shape[0] 
    zeroNum = zero_df.shape[0]
    zeroPrior = float(zeroNum / totalNum)
    return [zeroPrior, 1 - zeroPrior]


def read_test_file(fileName, freq, secLength):
    file_df = pd.read_csv(join(corpus_dir, fileName))
    # testFile_df = file_df.ix[(file_df['device'] == 'GENEActiv') & ((file_df[catagoryColName] == 1) | (file_df[catagoryColName] == 0))]
    testFile_df = file_df.ix[(file_df[catagoryColName] == 1) | (file_df[catagoryColName] == 0)]
    r, c = testFile_df.shape
    testList = []
    for i in range(r):
        pair = []
        X = read_X_single_file(i, testFile_df, freq, secLength)
        if not X:
            continue
        pair.append(X)
        # deal with y label
        pair.append(testFile_df.iloc[i][catagoryColName])

        testList.append(pair)

    return testList
    
maxLength = 0
minLength = 10000000
def read_X_single_file(rowNum, labelFile_df, freq, secLength):     
    xList = []
    for col in range(3):
        X = []
        xElement = openData(labelFile_df.iloc[rowNum]['dataFileHandleId_filepath'], col)
        if not xElement:
            return []
        xElement = np.asarray(xElement)
        if xElement.shape[0] > maxLength:
            maxLength = xElement.shape[0]
        if xElement.shape[0] < minLength:
            minLength = xElement.shape[0]
        
        # XElement = normalize(XElement)
        xElement = fourier_transform(xElement, freq)
        if sampleNormalize:
            xElement = normalize(xElement)
        r, c = xElement.shape

        xSection = cut_pieces(xElement, secLength)
        for x in xSection:
            if inputNormalize:
                x = normalize(x)
            X.append(x)
        x_batch = np.asarray(X)
        x_batch = x_batch.reshape(x_batch.shape[0], 1, x_batch.shape[1], x_batch.shape[2])
        xList.append(x_batch)
    return xList
    

def read_X_file_train(fileName, label, freq, secLength):
    file_df = pd.read_csv(join(corpus_dir, fileName))
    print(file_df.shape)
    # labelFile_df = file_df.ix[(file_df['device'] == 'GENEActiv') & (file_df[catagoryColName] == label)]
    labelFile_df = file_df.ix[file_df[catagoryColName] == label]
    X = []
    Y = []
    # labelLength : num of row, c: num of column
    labelLength, c = labelFile_df.shape

    xList = []
    for col in range(3):
        X = []
        for i in range(labelLength):
            xElement = openData(labelFile_df.iloc[i]['dataFileHandleId_filepath'], col)
            if not xElement:
                continue
            xElement = np.asarray(xElement)

            # XElement = normalize(XElement)
            xElement = fourier_transform(xElement, freq)
            # normalize
            if sampleNormalize:
                xElement = normalize(xElement)
            r, c = xElement.shape

            xSection = cut_pieces(xElement, secLength)
            for x in xSection:
                if inputNormalize:
                    x = normalize(x)
                X.append(x)
        
        x_batch = np.asarray(X)
        x_batch = x_batch.reshape(x_batch.shape[0], 1, x_batch.shape[1], x_batch.shape[2])
        xList.append(x_batch)
    return xList

# secLength means the length of the transformed section(time domain)
def cut_pieces(x_mag, secLength):
    # When input length < timeLength -> zero padding
    if(x_mag.shape[1] < secLength):
        temp = np.zeros((x_mag.shape[0], secLength)) 
        temp[:, 0 : x_mag.shape[1]] = x_mag
        x_mag = temp   
    freqBand, inputLength = x_mag.shape
    # cut pieces w.r.t timeLength. 
    # In the end, if not enough, extract the former frame 

    x_section = []
    sectionNum = math.floor(inputLength / secLength) + 1
    for i in range(sectionNum - 1):
        x_section.append(x_mag[:, i * secLength : (i+1) * secLength])

    # End
    # Last package length < timeLength, extract last num of timeLength to make spectrogram
    remainNum = inputLength - secLength * (sectionNum - 1)
    formerNum = secLength * (sectionNum - 1) - (secLength - remainNum)
    x_section.append(x_mag[:, formerNum :])

    return x_section
    
def cut_pieces_overlap(x_mag, secLength):
    # When input length < timeLength -> zero padding
    if(x_mag.shape[1] < secLength):
        temp = np.zeros((x_mag.shape[0], secLength)) 
        temp[:, 0 : x_mag.shape[1]] = x_mag
        x_mag = temp   
    freqBand, inputLength = x_mag.shape
    # cut pieces w.r.t timeLength. 
    # In the end, if not enough, extract the former frame 

    x_section = []
    cur = 0
    while math.floor((1 + cur * 0.5) * secLength) < inputLength:
        x_section.append(x_mag[:, math.floor(cur * 0.5 * secLength) : math.floor((cur * 0.5 + 1) * secLength)])
        cur += 1
    remainNum = inputLength - math.floor(cur * 0.5 * secLength)
    formerNum = math.floor(cur * 0.5 * secLength) - (secLength - remainNum)
    x_section.append(x_mag[:, formerNum :])
    return x_section

def normalize(arr):
    return arr/np.amax(arr)

def fourier_transform(x, freq):
    x_stft = librosa.core.stft(x, n_fft = freq)
    x = np.array(x_stft)
    x_mag , x_phs = librosa.core.magphase(x) 
    return x_mag

def openData(fileName, col):
    fid = open(fileName, 'r')
    X = []
    first = True 
    while True:
        dataLine = fid.readline().strip('\n') 
        if not dataLine:
            break
        if first:
            first = False
            continue
        dataLine = re.split('\s+', dataLine)
        if np.isfinite(float(dataLine[col + 1])): 
            X.append(float(dataLine[col + 1]))
    
    return X


if __name__ == '__main__':
    fileName = 'label_data.csv'
    # read_X_file(fileName, 1, 80, 43)
    # read_X_file(fileName, 0, 80, 43)
    # x_batch, y_batch = read_train_file(fileName, 80, 43)
    # print(x_batch.shape)
    # print(y_batch.shape)
    test_file = read_test_file('label_train_data.csv', 80, 43)
    print('max: ', maxLength)
    print('mix: ', minLength)
