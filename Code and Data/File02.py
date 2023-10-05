# load the important libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from astropy.table import Table
from sklearn.utils import shuffle

import warnings

# filter simple warning
warnings.simplefilter("ignore")


# load the dataset
data = pd.read_csv('data.csv')


# print the size of the data
print(data.shape)

# see the columns
print(data.columns)


# differentiating feature columns and target column
x = data[['Open','High','Low', 'Close', 'max', 'min']]
y = data['IsBreaking']

x=np.array(x)
y=np.array(y)

# feature scaling on attribute columns
ft_scl = preprocessing.StandardScaler()
ft_scl.fit(x)
ft_scl.transform(x)

# K (five) Fold Cross Validation 
kf = KFold(n_splits=5,  shuffle=True)
for train_index, test_index in kf.split(x):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # we create four lists to score accuracy , precision, recall and f1 scores
    accuracy_set=[]
    precision_set=[]
    recall_set=[]
    f1_score_set=[]



    # Apply k nearest neighbours classification method for classification
    knc = KNeighborsClassifier()
    knc.fit(x_train,y_train)
    prediction = knc.predict(x_test)
    accuracy_set.append(format(accuracy_score(y_test, prediction), ".4f"))
    precision_set.append(format(precision_score(y_test, prediction, average='macro'),".4f"))
    recall_set.append(format(recall_score(y_test, prediction, average='macro'),".4f"))
    f1_score_set.append(format(f1_score(y_test, prediction, average='macro'), ".4f"))


    # Apply decision tree classification method for classification
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    prediction = dtc.predict(x_test)
    accuracy_set.append(format(accuracy_score(y_test, prediction), ".4f"))
    precision_set.append(format(precision_score(y_test, prediction, average='macro'),".4f"))
    recall_set.append(format(recall_score(y_test, prediction, average='macro'),".4f"))
    f1_score_set.append(format(f1_score(y_test, prediction, average='macro'), ".4f"))
    

    # Apply random forest classification method for classification
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    prediction = rfc.predict(x_test)
    accuracy_set.append(format(accuracy_score(y_test, prediction), ".4f"))
    precision_set.append(format(precision_score(y_test, prediction, average='macro'),".4f"))
    recall_set.append(format(recall_score(y_test, prediction, average='macro'),".4f"))
    f1_score_set.append(format(f1_score(y_test, prediction, average='macro'), ".4f"))

    # Apply SGD classification method for classification
    sgd = SGDClassifier()
    sgd.fit(x_train,y_train)
    prediction = sgd.predict(x_test)
    accuracy_set.append(format(accuracy_score(y_test, prediction), ".4f"))
    precision_set.append(format(precision_score(y_test, prediction, average='macro'),".4f"))
    recall_set.append(format(recall_score(y_test, prediction, average='macro'),".4f"))
    f1_score_set.append(format(f1_score(y_test, prediction, average='macro'), ".4f"))

    # Apply AdaBoost classification method for classification
    adc = AdaBoostClassifier()
    adc.fit(x_train, y_train)
    prediction = adc.predict(x_test)
    accuracy_set.append(format(accuracy_score(y_test, prediction), ".4f"))
    precision_set.append(format(precision_score(y_test, prediction, average='macro'),".4f"))
    recall_set.append(format(recall_score(y_test, prediction, average='macro'),".4f"))
    f1_score_set.append(format(f1_score(y_test, prediction, average='macro'), ".4f"))

    # Apply MLP for classification
    mlp = MLPClassifier()
    mlp.fit(x_train,y_train)
    prediction = mlp.predict(x_test)
    accuracy_set.append(format(accuracy_score(y_test, prediction), ".4f"))
    precision_set.append(format(precision_score(y_test, prediction, average='macro'),".4f"))
    recall_set.append(format(recall_score(y_test, prediction, average='macro'),".4f"))
    f1_score_set.append(format(f1_score(y_test, prediction, average='macro'), ".4f"))

    # Apply SVM for classification
    svmClf = svm.SVC(decision_function_shape='ovo')
    svmClf.fit(x_train,y_train)
    prediction = svmClf.predict(x_test)
    accuracy_set.append(format(accuracy_score(y_test, prediction), ".4f"))
    precision_set.append(format(precision_score(y_test, prediction, average='macro'),".4f"))
    recall_set.append(format(recall_score(y_test, prediction, average='macro'),".4f"))
    f1_score_set.append(format(f1_score(y_test, prediction, average='macro'), ".4f"))

    
    

    # we create a table to show the accuracy, precision, recall and f1 score of the classification methods used
    t = Table()
    t['classification'] = ['KNN ','DT ','RF', 'SGD', 'AdaBoost', 'MLP', 'SVM']
    t['accuracy'] = accuracy_set
    t['precision'] = precision_set
    t['recall'] = recall_set
    t['f1 score'] = f1_score_set
    print(t)
