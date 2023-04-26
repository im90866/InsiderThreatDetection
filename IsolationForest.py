import pandas as pd
from pyod.models.iforest import IForest
import vaex

chunk_szie = 10_000_000

data = pd.read_csv("D:/Datasets/r4.2/r4.2/device.csv")
# data2 = vaex.from_csv("D:/Datasets/r4.2/r4.2/logon.csv", convert = True, chunk_size = 5_000_000)
# data3 = vaex.from_csv("D:/Datasets/r4.2/r4.2/http.csv", convert = True, chunk_size = 5_000_000)

# data2 = data2.rename(columns={"date": "Date2"})
# merged1 = data.join(data2, on="id")
# merged2 = data.join(data2, on="id")
# merged1.to_csv('merged_file.csv', index=False)

# features = [""]

# model = IForest()
# model.fit

# print(merged1.head())

#Skewness
data = data.drop('id', axis = 1)
data = data.drop('date', axis = 1)
skewness = data.skew(numeric_only=False)
print(skewness)