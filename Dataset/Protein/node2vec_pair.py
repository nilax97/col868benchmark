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

train_size = 10000

error = [21890,32320,34134,40901,41002]
line = open("PROTEINS_full_node_labels.txt").readlines()
n = len(line)
zero = dict()
one = dict()
two = dict()
count = np.zeros(3)
node_val = np.zeros(n)
for i in range(n):
    if i+1 in error:
        node_val[i] = -1
    else:
        node_val[i] = int(line[i])
        if(node_val[i]==0):
            zero[i] = 1
            count[0] +=1
        elif(node_val[i]==1):
            one[i] = 1
            count[1] +=1
        else:
            two[i] = 1
            count[2] +=1

total = np.sum(count)
train = np.zeros((train_size*4,3))
for i in range(train_size):
    ind = randint(0,total-1)
    if(ind<count[0]):
        x = random.choice(list(zero.keys()))
        y = random.choice(list(zero.keys()))
        a = random.choice(list(zero.keys()))
        b = random.choice(list(zero.keys()))
        train[i*4,0] = x
        train[i*4,1] = y
        train[i*4,2] = 1
        train[i*4+1,0] = a
        train[i*4+1,1] = b
        train[i*4+1,2] = 1
    elif(ind<count[1]+count[0]):
        x = random.choice(list(one.keys()))
        y = random.choice(list(one.keys()))
        a = random.choice(list(one.keys()))
        b = random.choice(list(one.keys()))
        train[i*4,0] = x
        train[i*4,1] = y
        train[i*4,2] = 1
        train[i*4+1,0] = a
        train[i*4+1,1] = b
        train[i*4+1,2] = 1
    else:
        x = random.choice(list(two.keys()))
        y = random.choice(list(two.keys()))
        a = random.choice(list(two.keys()))
        b = random.choice(list(two.keys()))
        train[i*4,0] = x
        train[i*4,1] = y
        train[i*4,2] = 1
        train[i*4+1,0] = a
        train[i*4+1,1] = b
        train[i*4+1,2] = 1
        
    ind = randint(0,total-1)
    if(ind<count[0]):
        x = random.choice(list(zero.keys()))
        y = random.choice(list(one.keys()))
        z = random.choice(list(two.keys()))
        train[i*4+2,0] = x
        train[i*4+2,1] = y
        train[i*4+2,2] = 0
        train[i*4+3,0] = x
        train[i*4+3,1] = z
        train[i*4+3,2] = 0
    elif(ind<count[1]+count[0]):
        x = random.choice(list(zero.keys()))
        y = random.choice(list(one.keys()))
        z = random.choice(list(two.keys()))
        train[i*4+2,0] = x
        train[i*4+2,1] = y
        train[i*4+2,2] = 0
        train[i*4+3,0] = y
        train[i*4+3,1] = z
        train[i*4+3,2] = 0
    else:
        x = random.choice(list(zero.keys()))
        y = random.choice(list(one.keys()))
        z = random.choice(list(two.keys()))
        train[i*4+2,0] = z
        train[i*4+2,1] = y
        train[i*4+2,2] = 0
        train[i*4+3,0] = x
        train[i*4+3,1] = z
        train[i*4+3,2] = 0

def agg_fun_1(x,y):
    return (x+y)/2
def agg_fun_2(x,y):
    return (x*y)
def agg_fun_3(x,y):
    return np.abs(x-y)
def agg_fun_4(x,y):
    return np.square(x-y)

dir_list = ["node2vec_embeddings/"]
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
        
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = LinearSVC(dual=False, class_weight='balanced')
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred)
            prec[index] = precision_score(y_test,pred)
            rec[index] = recall_score(y_test,pred)
            f1[index] = f1_score(y_test,pred)
            index += 1
        print("LinearSVC")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = ExtraTreesClassifier(class_weight='balanced', n_jobs=16)
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred)
            prec[index] = precision_score(y_test,pred)
            rec[index] = recall_score(y_test,pred)
            f1[index] = f1_score(y_test,pred)
            index += 1
        print("ExtraTreesClassifier")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = BaggingClassifier(n_jobs=16)
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred)
            prec[index] = precision_score(y_test,pred)
            rec[index] = recall_score(y_test,pred)
            f1[index] = f1_score(y_test,pred)
            index += 1
        print("BaggingClassifier")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestClassifier(class_weight='balanced', n_jobs=16)
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred)
            prec[index] = precision_score(y_test,pred)
            rec[index] = recall_score(y_test,pred)
            f1[index] = f1_score(y_test,pred)
            index += 1
        print("RandomForestClassifier")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = GradientBoostingClassifier()
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred)
            prec[index] = precision_score(y_test,pred)
            rec[index] = recall_score(y_test,pred)
            f1[index] = f1_score(y_test,pred)
            index += 1
        print("GradientBoostingClassifier")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
        
        
        roc = np.zeros(5)
        prec = np.zeros(5)
        rec = np.zeros(5)
        f1 = np.zeros(5)
        index = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = AdaBoostClassifier()
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
        
            roc[index] = roc_auc_score(y_test,pred)
            prec[index] = precision_score(y_test,pred)
            rec[index] = recall_score(y_test,pred)
            f1[index] = f1_score(y_test,pred)
            index += 1
        print("AdaBoostClassifier")
        print(round(np.average(roc),3))
        print(round(np.average(prec),3))
        print(round(np.average(rec),3))
        print(round(np.average(f1),3))
