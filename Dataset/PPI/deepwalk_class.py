import numpy as np
import json
import random
from random import randint
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import precision_score, recall_score,f1_score

class_map = json.load(open("ppi-class_map.json"))

n = len(class_map.keys())
m = len(class_map['0'])
target = np.zeros((n,m))
for i in range(n):
    temp = np.asarray(class_map[str(i)])
    target[i] = temp
del temp

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
                embedding[index,i] = float(val[i])
        X = embedding
        y = target

        roc = np.zeros(5*m)
        prec = np.zeros(5*m)
        rec = np.zeros(5*m)
        f1 = np.zeros(5*m)
        index = 0
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for i in range(m):
                clf = LogisticRegression(penalty='l2', solver='saga', class_weight='balanced', n_jobs=16)
                clf.fit(X_train,y_train[:,i])
                pred = clf.predict(X_test)
        
                roc[index*m+i] = roc_auc_score(y_test[:,i],pred)
                prec[index*m+i] = precision_score(y_test[:,i],pred)
                rec[index*m+i] = recall_score(y_test[:,i],pred)
                f1[index*m+i] = f1_score(y_test[:,i],pred)
            index += 1
        print("LogisticRegression")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))

        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = MLPClassifier(verbose=1)
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred, average='micro')
            prec[index] = precision_score(y_test,pred, average='micro')
            rec[index] = recall_score(y_test,pred, average='micro')
            f1[index] = f1_score(y_test,pred, average='micro')
            index += 1
        print("MLP")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = ExtraTreesClassifier(class_weight='balanced', n_jobs=20)
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred, average='micro')
            prec[index] = precision_score(y_test,pred, average='micro')
            rec[index] = recall_score(y_test,pred, average='micro')
            f1[index] = f1_score(y_test,pred, average='micro')
            index += 1
        print("ExtraTrees")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestClassifier(class_weight='balanced', n_jobs=20)
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred, average='micro')
            prec[index] = precision_score(y_test,pred, average='micro')
            rec[index] = recall_score(y_test,pred, average='micro')
            f1[index] = f1_score(y_test,pred, average='micro')
            index += 1
        print("RandomForest")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = DecisionTreeClassifier(class_weight='balanced')
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred, average='micro')
            prec[index] = precision_score(y_test,pred, average='micro')
            rec[index] = recall_score(y_test,pred, average='micro')
            f1[index] = f1_score(y_test,pred, average='micro')
            index += 1
        print("DecisionTrees")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = ExtraTreeClassifier(class_weight='balanced')
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred, average='micro')
            prec[index] = precision_score(y_test,pred, average='micro')
            rec[index] = recall_score(y_test,pred, average='micro')
            f1[index] = f1_score(y_test,pred, average='micro')
            index += 1
        print("ExtraTree (Non Ensemble)")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
