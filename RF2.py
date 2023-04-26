import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

http = pd.read_csv('D:/Datasets/r4.2/r4.2/http.csv')
email = pd.read_csv('D:/Datasets/r4.2/r4.2/email.csv')
file = pd.read_csv('D:/Datasets/r4.2/r4.2/file.csv')
logon = pd.read_csv('D:/Datasets/r4.2/r4.2/logon.csv')

data = pd.concat([http, email, file, logon])

data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

data['hour'] = data['date'].dt.hour
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

data = data.drop(['id', 'date', 'user', 'pc'], axis=1)

data = pd.get_dummies(data, columns=['activity'])

X_train, X_test, y_train, y_test = train_test_split(data.drop('threat', axis=1), data['threat'], test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))
