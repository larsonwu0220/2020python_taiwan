import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sheet = pd.read_excel(r'/home/ipcam-pc/Desktop/Nut/DataProcessing/MydataOkra.xls',sheet_name = 'Sheet1',skiprows=4)
attributes = sheet.columns.values.tolist()
print(attributes)

data = sheet.values
print(data.shape)
date = data[:-1,0]
MeanPrices = data[:-1,6]
print('date :',date)
print('Mean Prices :',MeanPrices)

plt.plot(date,MeanPrices)
plt.xticks(rotation=-90)
plt.show()
