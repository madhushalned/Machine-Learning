import pandas as pd

X1 = pd.read_csv('X_train.txt', header=None, delimiter=r"\s+" )
print("X1 Shape = ", X1.shape)

y1 = pd.read_csv('y_train.txt', header=None, delimiter=r"\s+" )
print("y1 Shape = ", y1.shape)

""" 
from sklearn.decomposition import PCA
Pca = PCA (n_components = 200)  
X1 = Pca.fit_transform(X1)   
print("X1 Shape (after PCA) = ", X1.shape) 
"""

X2 = pd.read_csv('X_test.txt', header=None, delimiter=r"\s+")
print("X2 Shape = ", X2.shape)

y2 = pd.read_csv('y_test.txt', header=None, delimiter=r"\s+")
print("y2 Shape = ", y2.shape)

"""
from sklearn.decomposition import PCA
Pca = PCA (n_components = 200)  
X2 = Pca.fit_transform(X2)  
print("X2 Shape (after PCA) = ", X2.shape) 
"""

import numpy
#Conversion of dataframe to 1D array
y1_new = y1.to_numpy()
y1 = y1_new.flatten()

print("\nShape of y1: ", y1.shape)

# Conversion of dataframe to 1D array
y2_new = y2.to_numpy()
y2 = y2_new.flatten()

print("Shape of y2: ", y2.shape)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X1, y1)

y_pred = classifier.predict(X2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y2)
print("\nConfusion Metrix: \n", cm)

from sklearn.metrics import accuracy_score
print("\nAccuracy: ", accuracy_score(y2, y_pred) * 100 )

from sklearn.metrics import precision_score
print("Precision: ", precision_score(y2, y_pred, average='weighted') * 100 )

from sklearn.metrics import recall_score
print("Recall: ", recall_score(y2, y_pred, average='weighted') * 100 )


#import matplotlib.pyplot as plt
#plt.scatter(X1, y1, marker='o', label='True Labels')
#plt.title('TRAINING DATA SET')
#plt.xlabel('Training data')
#plt.ylabel('Labelled data')
#plt.legend()
#plt.show()

#plt.scatter(X2, y2, marker='x', s=80, label='Prediction')
#plt.title('TESTING DATA SET')
#plt.xlabel('Testing data')
#plt.ylabel('Labelled data')
#plt.legend()
#plt.show()

print("\nGetting number of support vectors for each class: ")
print(classifier.n_support_)

print("\nGetting indices of support vectors: ")
print(classifier.support_)

print("\nGetting support vectors... \n")
print(classifier.support_vectors_)


