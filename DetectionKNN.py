import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

#Using to read datasets in chunks since the files are big
#and computational resources are not enough to run
chunkSize = 10**6

#Load the CERT Insider r6.2
device = pd.read_csv('D:/Datasets/r6.2/device.csv', chunksize=chunkSize)
# email = pd.read_csv('D:/Datasets/r6.2/email.csv')
file = pd.read_csv('D:/Datasets/r6.2/file.csv', chunksize=chunkSize)
http = pd.read_csv('D:/Datasets/r6.2/http.csv', chunksize=chunkSize)
# logon = pd.read_csv('D:/Datasets/r6.2/logon.csv')
# psychometric = pd.read_csv('D:/Datsets/r6.2/psychometric.csv')

httpChunks = []

for chunk in http:
    httpChunks.append(chunk)

httpDF = pd.concat(httpChunks, ignore_index= True)

def featureEng(chunk):
    chunk['date'] = pd.to_datetime(chunk['date'], format = '%m/%d/%Y %H:%M:%S')

    #do feature eng
    return 