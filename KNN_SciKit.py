# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:11:10 2023

@author: sam
"""

import matplotlib.pyplot as plt

#array for small test dataset
x = [3, 6, 11, 2, 5, 12, 15 , 7, 12, 11]
y = [20, 16, 23, 13, 15, 23, 27, 21, 20, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

#Using Scikit library for K nearest neighbours algorithm
from sklearn.neighbors import KNeighborsClassifier

#convert arrays to points
data = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=5)

#fit KNN model onto dataset using 5 nearest neighbours
knn.fit(data, classes)
new_x = 8
new_y = 21
new_point = [(new_x, new_y)]

prediction = knn.predict(new_point)

plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
