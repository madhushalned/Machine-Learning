import pandas as pd

X1 = pd.read_csv('X_train.txt', header=None, delimiter=r"\s+" )
#print("X1 Shape = ", X1.shape)

y1 = pd.read_csv('y_train.txt', header=None, delimiter=r"\s+" )
#print("y1 Shape = ", y1.shape)


X2 = pd.read_csv('X_test.txt', header=None, delimiter=r"\s+")
#print("X2 Shape = ", X2.shape)

y2 = pd.read_csv('y_test.txt', header=None, delimiter=r"\s+")
#print("y2 Shape = ", y2.shape)

## Conversion of dataframe to 1D array
y1_new = y1.to_numpy()
y1 = y1_new.flatten()

## Conversion of dataframe to 1D array
y2_new = y2.to_numpy()
y2 = y2_new.flatten()

from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
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

from matplotlib import pyplot as plt
from sklearn import tree

text_representation = tree.export_text(classifier)
print(text_representation)
plt.figure(figsize=(8, 8))
tree.plot_tree(classifier, filled=True, feature_names=X1)
plt.show()

tree.plot_tree(classifier)
