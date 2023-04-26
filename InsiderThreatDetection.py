import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

chunkSize = 10**6

#import csv files, i.e. CERT insider threat datasets
device = pd.read_csv('D:/Datasets/r6.2/device.csv', chunksize=chunkSize)
# email = pd.read_csv('D:/Datasets/r6.2/email.csv')
file = pd.read_csv('D:/Datasets/r6.2/file.csv', chunksize=chunkSize)
http = pd.read_csv('D:/Datasets/r6.2/http.csv', chunksize=chunkSize)
# logon = pd.read_csv('D:/Datasets/r6.2/logon.csv')
# psychometric = pd.read_csv('D:/Datsets/r6.2/psychometric.csv')

modelLR = LogisticRegression(max_iter=1000)
scalerLR = StandardScaler()

def process(httpChunk):
    httpChunk['date'] = pd.to_datetime(httpChunk['date'], format = '%m/%d/%Y %H:%M:%S')
    cols = ['pc', 'url', 'activity']
    for col in cols:
        le = LabelEncoder()
        httpChunk[col] = le.fit_transform(httpChunk[col].astype(str))

    httpChunk.fillna(httpChunk.mean(), inplace=True)

    X = httpChunk.drop(['id', 'date', 'user', 'content'], axis=1)
    y = httpChunk[['insider']]

    return X, y

res =[]
for deviceChunk in device:
    # deviceChunk['insider'] = np.where(deviceChunk['url'].str.contains('wikileaks.org'), 1, 0)
    # res = device[httpChunk['url'].str.contains('wikileaks.org')]

    deviceChunk['date'] = pd.to_datetime(deviceChunk['date'], format = '%m/%d/%Y %H:%M:%S')
    deviceChunk['insider'] = np.where((deviceChunk['date'].dt.hour > 17) | (deviceChunk['date'].dt.hour < 9) & (deviceChunk['activity'] == 'Connect'), 1, 0)

    print(deviceChunk)
    cols = ['pc', 'file_tree', 'activity']
    for col in cols:
        le = LabelEncoder()
        deviceChunk[col] = le.fit_transform(deviceChunk[col].astype(str))

    deviceChunk.fillna(deviceChunk.mean(), inplace=True)

    X = deviceChunk.drop(['id', 'date', 'user'], axis=1)
    y = deviceChunk[['insider']]

    trainData, testData = train_test_split(deviceChunk, test_size= 0.2, random_state= 42)
    xTrain = deviceChunk.drop(['id', 'date', 'user'], axis=1)
    yTrain = deviceChunk[['insider']]
    xTest = deviceChunk.drop(['id', 'date', 'user'], axis=1)
    yTest = deviceChunk[['insider']]
    # xChunk, yChunk = process(deviceChunk)
    modelLR.fit(xTrain, yTrain.values.ravel())
    yPred = modelLR.predict(xTest)
    acc = accuracy_score(yTest, yPred)
    conf = confusion_matrix(yTest, yPred)
    prec = precision_score(yTest, yPred)
    rec = recall_score(yTest, yPred)
    print('Accuracy', acc)
    print('Confusion', conf)
    print('Precision', prec)
    print('Recall score', rec)


# xTest, yTest = process(testData)



# scoreLR = clf.score(xStd, y)
# print('Accuracy: ', scoreLR)

#Function to perform feature engineering
# def afterHours():
#     device['date'] = pd.to_datetime(device['date'], format = '%m/%d/%Y %H:%M:%S')

#     device['date'].dt.hour > 17 & device['activity'] == 'Connect' & file['activity'] == 'File Open'

