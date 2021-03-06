import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def distance_euclid(p1, p2):
    dst = np.sqrt(((p1-p2)**2).sum())
    return dst

def KNN(X_Train, Y_Train, X_Test, k=5):

    vals = []
    for ix in range(X_Train.shape[0]):
        dst = distance_euclid(X_Train[ix], X_Test)
        vals.append([dst, Y_Train[ix]])

    sorted_vals = sorted(vals, key=lambda mn:mn[0])
    neighbors = np.array(sorted_vals)[:k, -1]
    freq = np.unique(neighbors, return_counts=True)
    my_ans = freq[0][freq[1].argmax()]
    return my_ans
ds = pd.read_csv('fashion-mnist_train.csv')
data = ds.values[:3000,:]

split = int(data.shape[0]*0.80)

X_Train = data[:split, 1:]
Y_Train = data[:split, 0]

X_Test = data[split: , 1:]
Y_Test = data[split:, 0]

ans = KNN(X_Train, Y_Train, X_Test[0])
plt.imshow(X_Test[0].reshape(28,28), cmap='gray')
plt.show()
print ans
