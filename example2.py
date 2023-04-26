import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Read the logon.csv file
logon_data = pd.read_csv('D:/Datasets/r4.2/r4.2/logon.csv')

# Preprocessing
logon_data['date'] = pd.to_datetime(logon_data['date'])
logon_data['hour'] = logon_data['date'].dt.hour

# Set working hours as 9 AM to 5 PM
working_hours_start = 9
working_hours_end = 17

# Create target variable: 1 if logon is outside working hours, 0 if within working hours
logon_data['outside_working_hours'] = ((logon_data['hour'] < working_hours_start) | (logon_data['hour'] >= working_hours_end)).astype(int)

# Feature extraction
features = ['hour']
X = logon_data[features]
y = logon_data['outside_working_hours']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print('\nAccuracy Score:')
print(accuracy_score(y_test, y_pred))
