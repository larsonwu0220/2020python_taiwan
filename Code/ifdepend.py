import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr

sheet1 = pd.read_excel(r'/home/ipcam-pc/Desktop/Nut/DataProcessing/MydataOkra.xls',sheet_name = 'Sheet1',skiprows=4)
sheet2 = pd.read_excel(r'/home/ipcam-pc/Desktop/Nut/DataProcessing/MydataLoofah.xls',sheet_name = 'Sheet1',skiprows=4)

DataOkra = sheet1.values
DataLoofah = sheet2.values
DataOkra_date = DataOkra[:-1,0]
DataLoofah_date = DataLoofah[:-1,0]

DataOkra_count = DataOkra[:-1,8]
DataLoofah_count = DataLoofah[:-1,8]

# plt.plot(DataOkra_date,DataOkra_count)
# plt.plot(DataOkra_date,DataLoofah_count,'r')
# plt.xticks(rotation=-90)
#plt.show()

def mynormalization(x):
    max_x = np.max(x)
    x_normal = x/max_x
    return x_normal

def MyPearsons(x,y):
    # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    x = np.asarray(x,dtype=np.float32)
    y = np.asarray(y,dtype=np.float32)
    n = len(x)
    xmean = np.mean(x)
    ymean = np.mean(y)
    a= np.sum(((x-xmean)/np.std(x))*((y-ymean)/np.std(y)))
    return a/(n)

x_okra = mynormalization(DataOkra_count)
x_loofah = mynormalization(DataLoofah_count)
plt.plot(DataOkra_date,x_okra)
plt.plot(DataOkra_date,x_loofah,'r')
plt.xticks(rotation=-90)
#plt.show()
corr, _ = pearsonr(x_okra, x_loofah)
print('Pearsons correlation: %.3f' % corr)

mycorr = MyPearsons(x_okra, x_loofah)
print('My Pearsons correlation: %.3f' % mycorr)

corr2, _ = spearmanr(x_okra, x_loofah)
print('Spearmans correlation: %.3f' % corr2)