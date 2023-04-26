import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logon = pd.read_csv('D:/Datasets/r6.2/logon.csv')
device = pd.read_csv('D:/Datasets/r6.2/device.csv')

logonSample = logon.sample(n=100000, random_state= 42)
deviceSample = device.sample(n=100000, random_state= 42)

mergedData = pd.merge(logonSample, deviceSample, on=['user'])
print(mergedData.shape)

encoder = LabelEncoder()
mergedData['user'] = encoder.fit_transform(mergedData['user'])
mergedData['pc_x'] = encoder.fit_transform(mergedData['pc_x'])
mergedData['pc_y'] = encoder.fit_transform(mergedData['pc_y'])
mergedData['activity_x'] = encoder.fit_transform(mergedData['activity_x'])
mergedData['activity_y'] = encoder.fit_transform(mergedData['activity_y'])

mergedData['date_x'] = pd.to_datetime(mergedData['date_x'], format = '%m/%d/%Y %H:%M:%S')
mergedData['date_y'] = pd.to_datetime(mergedData['date_y'], format = '%m/%d/%Y %H:%M:%S')
mergedData['hour'] = mergedData['date_y'].dt.hour
mergedData['insider'] = np.where((mergedData['hour'] > 17) | (mergedData['hour'] < 8) & (mergedData['activity_y'] == 'Connect'), 1, 0)

print(mergedData.head())
features = mergedData[['user', 'hour', 'activity_x', 'activity_y']]
target = mergedData['insider'] 

xTrain, xTest, yTrain, yTest = train_test_split(features, target, test_size=0.2, random_state=42)

# lrModel = LogisticRegression(solver='liblinear')
# lrModel.fit(xTrain, yTrain)


# yPred = lrModel.predict(xTest)
# confusionMatrix = confusion_matrix(yTest, yPred)

# print(classification_report(yTest, yPred))
# print(confusionMatrix)

# plt.figure(figsize=(10, 7))
# sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

rfModel = RandomForestClassifier(n_estimators=100, random_state=42)
rfModel.fit(xTest, yTrain)

yPred = rfModel.predict(xTest)
yPredProb = rfModel.predict_proba(xTest)[:, 1]
accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
roc_auc = roc_auc_score(yTest, yPredProb)
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')
print(f'ROC AUC: {roc_auc:.3f}')