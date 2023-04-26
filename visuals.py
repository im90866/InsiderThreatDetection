import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

numColumns = mergedData.select_dtypes(include='number')
skew = numColumns.skew()
print(skew)
plt.figure(figsize=(12, 6))
sns.barplot(x=skew.index, y=skew)
plt.xlabel('Column Name')
plt.ylabel('Skewness')
plt.title('Skewness of Numeric Columns in CSV file')
plt.xticks(rotation=90)
plt.show()