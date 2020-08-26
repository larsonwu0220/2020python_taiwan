from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

problem = 1

if problem == 1:
    #X, y = make_classification(n_samples=100, random_state=1)

    X, y = make_classification(n_samples=1000,n_features=2, n_redundant=0, n_informative=1,n_clusters_per_class=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
    ## X_train : (75,2) y_train : (75,)
    plt.scatter(X_test[:,0],X_test[:,1])
    plt.show()
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    y=clf.predict_proba(X_test[1:2,:])
    print(y)
    y=clf.predict(X_test[1:2,:])
    print(y)

if problem == 2:
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    pred=clf.predict(X_test)
    Mat = np.zeros((10,10))
    for i in range(len(pred)):
        t = y_test[i]
        p = pred[i]
        Mat[t,p] += 1
    print(Mat)

    #print(images_and_labels[80])
    one_data = images_and_labels[8]
    img = one_data[0]
    real_label = one_data[1]
    img_64 = img.reshape((64, -1))
    mypred=clf.predict(img_64.T)
    print('pred : '+str(mypred) + ', real : '+str(real_label))
