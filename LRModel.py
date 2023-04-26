import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("D:/Datasets/r4.2/r4.2/device.csv")

data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M')

data['label'] = data['activity'].apply(lambda x: 0 if x == 'Disconnect' else 1)

data = pd.get_dummies(data, columns=['user', 'pc'])

data.drop(['id', 'activity'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))