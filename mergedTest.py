import pandas as pd

# Read the CSV files
logon_df = pd.read_csv('D:/Datasets/r4.2/r4.2/logon.csv', nrows= 1000)
device_df = pd.read_csv('D:/Datasets/r4.2/r4.2/device.csv', nrows= 1000)
email_df = pd.read_csv('D:/Datasets/r4.2/r4.2/email.csv', nrows= 1000)
file_df = pd.read_csv('D:/Datasets/r4.2/r4.2/file.csv', nrows= 1000)
http_df = pd.read_csv('D:/Datasets/r4.2/r4.2/http.csv', nrows= 1000)

# Preprocessing and Feature Engineering

# Logon and logoff patterns
logon_df['datetime'] = pd.to_datetime(logon_df['date'] + ' ' + logon_df['time'])
logon_df['hour'] = logon_df['datetime'].dt.hour
logon_df['day_of_week'] = logon_df['datetime'].dt.dayofweek
logon_df['is_weekend'] = logon_df['day_of_week'].isin([5, 6]).astype(int)
logon_df['is_failed'] = (logon_df['activity'] == 'Logoff').astype(int)

# Device usage
device_df['datetime'] = pd.to_datetime(device_df['date'] + ' ' + device_df['time'])
device_df['hour'] = device_df['datetime'].dt.hour
device_df['day_of_week'] = device_df['datetime'].dt.dayofweek
device_df['is_weekend'] = device_df['day_of_week'].isin([5, 6]).astype(int)

# Email activity
email_df['datetime'] = pd.to_datetime(email_df['date'] + ' ' + email_df['time'])
email_df['hour'] = email_df['datetime'].dt.hour
email_df['day_of_week'] = email_df['datetime'].dt.dayofweek
email_df['is_weekend'] = email_df['day_of_week'].isin([5, 6]).astype(int)
email_df['attachment_size'] = email_df['size'].apply(lambda x: int(x.split(' ')[0]))
email_df['is_external'] = email_df['to'].apply(lambda x: 1 if '@' in x else 0)

# File activity
file_df['datetime'] = pd.to_datetime(file_df['date'] + ' ' + file_df['time'])
file_df['hour'] = file_df['datetime'].dt.hour
file_df['day_of_week'] = file_df['datetime'].dt.dayofweek
file_df['is_weekend'] = file_df['day_of_week'].isin([5, 6]).astype(int)

# HTTP activity
http_df['datetime'] = pd.to_datetime(http_df['date'] + ' ' + http_df['time'])
http_df['hour'] = http_df['datetime'].dt.hour
http_df['day_of_week'] = http_df['datetime'].dt.dayofweek
http_df['is_weekend'] = http_df['day_of_week'].isin([5, 6]).astype(int)

# Merge the dataframes
# You may need to decide on how to merge the data, depending on the problem context and the specific features you want to use.
merged_df = logon_df.merge(device_df, on=['datetime', 'user', 'pc'], how='outer')
merged_df = merged_df.merge(email_df, on=['datetime', 'user', 'pc'], how='outer')
merged_df = merged_df.merge(file_df, on=['datetime', 'user', 'pc'], how='outer')
merged_df = merged_df.merge(http_df, on=['datetime', 'user', 'pc'], how='outer')

# Fill missing values
merged_df.fillna(0, inplace=True)

# Save the preprocessed and merged dataframe to a new CSV file
merged_df.to_csv('merged_preprocessed.csv', index=False)
