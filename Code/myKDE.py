import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

sheet1 = pd.read_excel(r'/home/ipcam-pc/Desktop/Nut/DataProcessing/Grapefruit.xls',sheet_name = 'Sheet1',skiprows=4)

myData = sheet1.values

myPrice = myData[:-1,6]


def mynormalization(x):
    max_x = np.max(x)
    x_normal = x/max_x
    return max_x,x_normal

max_x, x_data = mynormalization(myPrice)
kde = KernelDensity(bandwidth=0.02, kernel='gaussian')
kde.fit(x_data[:, None])
xx = np.linspace(0,1,101)
yy = kde.score_samples(xx[:, None]) ## log values
yy = np.exp(yy)
plt.plot(xx,yy,'r')
plt.plot(x_data, np.zeros(len(x_data)),'.')
plt.show()


current_price = 12
x_test = np.array([current_price/max_x])
x_test = x_test[:,None]
x_test_prob = np.exp(kde.score_samples(x_test))
print('prob :',x_test_prob[0])