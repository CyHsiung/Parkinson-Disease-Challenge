#auther: Xinlin Song (xinlins@umich.edu)
import synapseclient
import json
import pandas as pd
import numpy as np

#login synapse account. Put your user name and passowrd in login function: syn=synapseclient.login('me@umich.edu', 'password')
syn=synapseclient.login('cyhsiung1994@gmail.com', 'peter321')

#Synapse ID of walking activity data and demographic data, you need to change the table IDs to the table you want to download
LDOPA_TRAIN_SYNID = "syn10495809" 
LDOPA_TEST_SYNID = "syn10701954"

#remove LIMIT 1 to dowonload the the whole data set. Use LIMIT 1 to test whether the code works.
#download the data frame (not json file is downloaded)
train_results = syn.tableQuery(('SELECT * FROM {0}').format(LDOPA_TRAIN_SYNID))
df_train_results=train_results.asDataFrame()
#Save the demo data to csv file


#download the json files

train_json_files=syn.downloadTableColumns(train_results, 'dataFileHandleId')
#get the fileID and the file path
items=train_json_files.items()

train_json_files_temp = pd.DataFrame({'dataFileHandleId': [i[0] for i in items], 'dataFileHandleId_filepath': [i[1] for i in items]})

#Change the fileID type to float for future merge
train_json_files_temp['dataFileHandleId']=train_json_files_temp['dataFileHandleId'].astype(int)
df_train_results['dataFileHandleId']=df_train_results['dataFileHandleId'].astype(int)

#merge json file path with the main dataframe
df_train_results=df_train_results.merge(train_json_files_temp, how='left', on='dataFileHandleId')
df_train_results.to_csv('./corpus/train_data.csv', index=False)

# Download test data
#remove LIMIT 1 to dowonload the the whole data set. Use LIMIT 1 to test whether the code works.
#download the data frame (not json file is downloaded)
test_results = syn.tableQuery(('SELECT * FROM {0}').format(LDOPA_TEST_SYNID))
df_test_results=test_results.asDataFrame()
#Save the demo data to csv file

#download the json files

test_json_files=syn.downloadTableColumns(test_results, 'dataFileHandleId')
#get the fileID and the file path
items=test_json_files.items()

test_json_files_temp = pd.DataFrame({'dataFileHandleId': [i[0] for i in items], 'dataFileHandleId_filepath': [i[1] for i in items]})

#Change the fileID type to int for future merge
test_json_files_temp['dataFileHandleId']=test_json_files_temp['dataFileHandleId'].astype(int)
df_test_results['dataFileHandleId']=df_test_results['dataFileHandleId'].astype(int)

#merge json file path with the main dataframe
df_test_results=df_test_results.merge(test_json_files_temp, how='left', on='dataFileHandleId')
df_test_results.to_csv('./corpus/test_data.csv', index=False)
