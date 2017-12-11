import pandas as pd
import sys


def count_file_row(fileName):
    file_df = pd.read_csv(fileName)
    r, c = file_df.shape
    print(r)

if __name__ == '__main__':
    fileName = sys.argv[1]
    count_file_row(fileName)

