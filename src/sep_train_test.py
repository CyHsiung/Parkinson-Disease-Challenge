import pandas as pd
import numpy as np
from os.path import basename, splitext, join
import re
import math
import random


project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
feats_dir = join(project_dir, "feats")

def separate_train_test_random(fileName, ratio):
    file_df = pd.read_csv(join(corpus_dir, fileName))
    r, c = file_df.shape
    train_df = file_df[0:1]
    test_df = file_df[1:2]
    for i in range(2, r):
        if (random.random() > ratio):
            test_df = test_df.append(file_df.ix[i])
        else:
            train_df = train_df.append(file_df.ix[i])
    print(train_df.shape)
    print(test_df.shape)
    train_df.to_csv(join(corpus_dir, 'label_train_data_train.csv'))
    test_df.to_csv(join(corpus_dir, 'label_test_data_valid.csv'))

def separate_train_test_regular(fileName, ratio):  
    file_df = pd.read_csv(join(corpus_dir, fileName))
    r, c = file_df.shape
    train_df = file_df[0 : math.floor(ratio * r)]
    test_df = file_df[math.floor(ratio * r) :]
    train_df.to_csv(join(corpus_dir, 'label_train_data_train.csv'))
    test_df.to_csv(join(corpus_dir, 'label_train_data_valid.csv'))

if __name__ == '__main__':
    fileName = 'label_train_data.csv'
    ratio = 0.85

    separate_train_test_random(fileName, ratio)

