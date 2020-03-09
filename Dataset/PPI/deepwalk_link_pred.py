import numpy as np
import random
from random import randint
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import precision_score, recall_score,f1_score

n = 56944
line = open("ppi-edge-list.txt").readlines()
m = len(line)
train = np.zeros((2*m,3))
edges = dict()
for i in range(m):
    temp = line[i].strip().replace("\"","").split()
    temp[0] = int(temp[0])
    temp[1] = int(temp[1])
    x = min(temp[0],temp[1])
    y = max(temp[0],temp[1])
    train[i][0] = x
    train[i][1] = y
    train[i][2] = 1
    edges[(x,y)] = 1

index = 0
while(index<m):
    x = randint(0,n-1)
    y = randint(0,n-1)
    if(x==y):
        continue
    a = min(x,y)
    b = max(x,y)
    if (a,b) in edges:
        continue
    train[m+index][0] = a
    train[m+index][1] = b
    train[m+index][2] = 0
    index += 1

np.random.shuffle(train)

def agg_fun_1(x,y):
    return (x+y)/2
def agg_fun_2(x,y):
    return (x*y)
def agg_fun_3(x,y):
    return np.abs(x-y)
def agg_fun_4(x,y):
    return np.square(x-y)

dir_list = ["deepwalk_embeddings/"]
# dir_list = ["node2vec_embeddings/"]

for folder in dir_list:
    for file in os.listdir(folder):
        print(file)
        emb = open(os.path.join(folder,file)).readlines()[1:]
        size = len(emb[0].strip().split())
        embedding = np.zeros((n,size-1))
        indlist = dict()
        for x in emb:
            temp = x.strip().split()
            index = int(temp[0])
            indlist[index] = 1
            val = temp[1:]
            for i in range(len(val)):
                embedding[index-1,i] = float(val[i])
        X = np.zeros((train.shape[0],size-1))
        y = np.zeros(train.shape[0])
        for i in range(train.shape[0]):
            X[i] = agg_fun_1(embedding[int(train[i,0])],embedding[int(train[i,1])])
            y[i] = train[i,2]
            
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = LogisticRegression(penalty='l2', solver='saga',class_weight='balanced')
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred)
            prec[index] = precision_score(y_test,pred)
            rec[index] = recall_score(y_test,pred)
            f1[index] = f1_score(y_test,pred)
            index += 1
        print("LogisticRegression")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
