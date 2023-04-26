import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

http_df = pd.read_csv("D:/Datasets/r4.2/r4.2/http.csv")
file_df = pd.read_csv("D:/Datasets/r4.2/r4.2/file.csv")
device_df = pd.read_csv("D:/Datasets/r4.2/r4.2/device.csv")
email_df = pd.read_csv("D:/Datasets/r4.2/r4.2/email.csv")

df = pd.concat([http_df, file_df, device_df, email_df])
df = df.drop(columns=['id', 'date', 'user', 'pc'])
df = pd.get_dummies(df, columns=['activity'])

y = df['target']
X = df.drop(columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))
