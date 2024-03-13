#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn import metrics
#load the dataset
df = pd.read_csv('Social_Network_Ads.csv')
#splitting the dataset into independent and dependent variables
X = df.iloc[:, [2,3]].values
y = df.iloc[:, 4].values
#split X and y into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=0)
#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#fitting the SVM
clf = SVC(kernel ='rbf', random_state=0) #building Kernal SVM model 
clf.fit(X_train, y_train)
#Prediction
y_pred = clf.predict(X_test)
#finding the accuracy
acc = accuracy_score(y_test, y_pred)
print("Acc", acc*100)
#confusion matrix 
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix for purchased data', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()