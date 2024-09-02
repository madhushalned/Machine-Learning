import pandas as pd

X1 = pd.read_csv('X_train.txt', header=None, delimiter=r"\s+" )
#print("X1 Shape = ", X1.shape)

y1 = pd.read_csv('y_train.txt', header=None, delimiter=r"\s+" )
#print("y1 Shape = ", y1.shape)

X2 = pd.read_csv('X_test.txt', header=None, delimiter=r"\s+")
#print("X2 Shape = ", X2.shape)

y2 = pd.read_csv('y_test.txt', header=None, delimiter=r"\s+")
#print("y2 Shape = ", y2.shape)

# Conversion of dataframe to 1D array
y1_new = y1.to_numpy()
y1 = y1_new.flatten()

# Conversion of dataframe to 1D array
y2_new = y2.to_numpy()
y2 = y2_new.flatten()

from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=7) 
classifier.fit(X1, y1)

y_pred = classifier.predict(X2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y2)
print(cm)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y2, y_pred) * 100 )

from sklearn.metrics import precision_score
print("Precision:", precision_score(y2, y_pred, average='weighted') * 100 )

from sklearn.metrics import recall_score
print("Recall:", recall_score(y2, y_pred, average='weighted') * 100 )

