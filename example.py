import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# # Replace 'file1.csv' and 'file2.csv' with the appropriate file names
# data = pd.read_csv("D:/Datasets/r4.2/r4.2/device.csv")
# data2 = pd.read_csv("D:/Datasets/r4.2/r4.2/email.csv")
# data3 = pd.read_csv("D:/Datasets/r4.2/r4.2/file.csv")

# logon = pd.read_csv('D:/Datasets/r6.2/logon.csv')
# device = pd.read_csv('D:/Datasets/r6.2/device.csv')
# file = pd.read_csv('D:/Datasets/r6.2/file.csv')
# http = pd.read_csv('D:/Datasets/r6.2/http.csv')

# logon = logon.drop('id', axis=1)
# device = device.drop('id', axis=1)

# device['date'] = pd.to_datetime(device['date'], format = '%m/%d/%Y %H:%M:%S')
# logon['date'] = pd.to_datetime(logon['date'], format = '%m/%d/%Y %H:%M:%S')

# logonUser = logon.loc[(logon['user'] == 'ACM2278') & (logon['date'].dt.hour > 16)]
# deviceUser = device.loc[
#     (device['user'] == 'ACM2278') & 
#     (device['date'].dt.hour > 16) &
#     (device['activity'] == 'Logon')
#     ]

# httpWiki = http[http['url'].str.contains('wikileaks.org')]
# merged_data = logonUser.merge(deviceUser, on=['user'])
# merged_data.fillna(method='ffill', inplace=True)

# merged_data.to_csv('test.csv', index=False)
# httpWiki.to_csv('test.csv', index=False)

