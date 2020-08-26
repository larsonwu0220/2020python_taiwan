import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

sheet1 = pd.read_excel(r'/home/ipcam-pc/Desktop/Nut/DataProcessing/MydataOkra.xls',sheet_name = 'Sheet1',skiprows=4)
sheet2 = pd.read_excel(r'/home/ipcam-pc/Desktop/Nut/DataProcessing/MydataLoofah.xls',sheet_name = 'Sheet1',skiprows=4)
DataOkra = sheet1.values
DataLoofah = sheet2.values
DataOkra_count = DataOkra[:-1,6]
DataLoofah_count = DataLoofah[:-1,6]

def mynormalization(x):
    max_x = np.max(x)
    x_normal = x/max_x
    return max_x, x_normal

okra_max, x_okra = mynormalization(DataOkra_count)
loofah_max, x_loofah = mynormalization(DataLoofah_count)

values = np.vstack([x_okra, x_loofah]).T # dim : [N,d]
kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
kde.fit(values)

X, Y = np.mgrid[0:1:101j, 0:1:101j]
positions = np.vstack([X.ravel(), Y.ravel()])

yy = kde.score_samples(positions.T)
yy = np.exp(yy)
Z = np.reshape(yy.T, X.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[0, 1, 0, 1])
ax.plot(x_okra, x_loofah, 'k.', markersize=1)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
plt.show()
