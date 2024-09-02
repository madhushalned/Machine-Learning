import pandas as pd

X1 = pd.read_csv('X_train.csv')
y1 = pd.read_csv('y_train.csv')



X2 = pd.read_csv('X_test.txt')
y2 = pd.read_csv('y_test.txt')


"""

missing_values = data1.isnull().sum()
missing_values = data2.isnull().sum()

STEP 0 : 
https://machinelearningmastery.com/evaluate-machine-learning-algorithms-for-human-activity-recognition/
upload/import csv file in google colab
from google.colab import files
uploaded = files.upload()


STEP1 : 
import pandas as pd

DATA = pd.read_csv('/content/drive/My Drive/path/to/your/csv/iris.csv')

STEP 2 : 
len(data)
df.size
df.shape
missing_values = data.isnull().sum()

//from sklearn.preprocessing import LabelEncoder
//label_encoder = LabelEncoder()

data = data.dropna()

data['column_name'].fillna(data['column_name'].mean(), inplace=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
model=scaler.fit(data)
scaled_data=model.transform(data)


from sklearn.preprocessing import StandardScaler
# compute required values
scaler = StandardScaler()
model = scaler.fit(data)
scaled_data = model.transform(data)


STEP 3 : ==> CLASSIFICATION PROBLEM  : no. of classes = 5 ?

STEP 4 : 
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


STEP 5 : 

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(X, Y, n_neighbors=3)
knn.fit(data, classes)


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)


from sklearn.ensemble import RandomForestClassifier  
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier.fit(x_train, y_train)  


STEP 6 : // Predict the Test Set Results

y_pred = classifier.predict(X_test)


STEP 7 :   // model evaluation


//TO DISPLAY TP, TN, FP, FN
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn import metrics

print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100 )

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))




"""



