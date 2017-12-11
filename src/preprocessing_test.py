import pandas as pd
import sys
from os.path import basename, splitext, join

corpus_dir = '../corpus'

def change_score_to_minus_one(fileName, feature):
    file_df = pd.read_csv(fileName)
    r, c = file_df.shape
    for i in range(len(file_df)):
        if file_df.iloc[i][feature] == 'Score':
            file_df.set_value(i, feature, -1)
        else:
            file_df.set_value(i, feature, -1)
    return file_df
            

if __name__ == '__main__':
    fileName = sys.argv[1]
    feature = 'bradykinesiaScore'
    file_df = change_score_to_minus_one(fileName, feature)
    file_df.to_csv(join(corpus_dir,'test_data_' + feature + '.csv'), index = False)
    
