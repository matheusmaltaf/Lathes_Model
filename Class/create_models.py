from lathes_model import LathesModel
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

input_list = [1,2,3,4,5,6]

gra_list = [1,2,3,4]

classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma='scale'),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1,max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

folds = 5

for input_id in input_list:
    PATH = '../Input/Input_%i.csv' %(input_id)

    full_data = np.genfromtxt(PATH,delimiter=',')

    n_measures = int(full_data[:,1].max())
    n_timeseries = int(full_data[:,0].max())
              
    unique_id = full_data[::n_measures,0]
    target = full_data[::n_measures,-1]

    kf = StratifiedKFold(n_splits=folds, random_state=12, shuffle=True)

    model = LathesModel()

    for it,(train_unique_index, test_unique_index) in enumerate(kf.split(unique_id, target)):
        print('-----> Input -', input_id, '\n-----> Fold -' , it, '\n', datetime.now())
        
        train_unique_index.sort()
        test_unique_index.sort()
        
        L_train = train_unique_index.shape[0]
        train_index = np.zeros(L_train*n_measures, dtype=np.int32)
        for ii in range(L_train):
            train_index[ii*n_measures:(ii+1)*n_measures] = list(range(train_unique_index[ii]*n_measures,
                                                                    (train_unique_index[ii]+1)*n_measures)) 
        L_test = test_unique_index.shape[0]
        test_index = np.zeros(L_test*n_measures, dtype=np.int32)
        for ii in range(L_test):
            test_index[ii*n_measures:(ii+1)*n_measures] = list(range(test_unique_index[ii]*n_measures,
                                                                    (test_unique_index[ii]+1)*n_measures)) 
        
        X_train = full_data[train_index,:-1]
        y_train = full_data[train_index,-1]

        X_test = full_data[test_index,:-1]
        y_test = full_data[test_index,-1]
        y_test = y_test[::n_measures]
        Classifiers_result = {}

        for gra in gra_list:
            Accuracy = np.zeros((len(classifiers) + 1))
            Precision = np.zeros((len(classifiers)))
            Recall = np.zeros((len(classifiers)))
            F1 = np.zeros((len(classifiers)))

            Classifiers_result[gra] = {'Nearest Neighbors':{},
                        'Linear SVM':{},
                        'RBF SVM':{},
                        'Gaussian Process':{},
                        'Decision Tree':{},
                        'Random Forest':{},
                        'Neural Net':{},
                        'AdaBoost':{},
                        'Naive Bayes':{},
                        'QDA':{}}

            for name, clf in zip(names,classifiers):
                Classifiers_result[gra][name] = {'Accuracy': 0,
                                    'Precision':0,
                                    'Recall':0,
                                    'F1':0,
                                    'time':0}

                params = {'granularity': gra,
                          'clf': clf}

                model.change_hyperparams(params)

                model.fit_after_tsfresh(X_train, y_train)

                if model.one_class_:
                    Classifiers_result[gra]= 'Error: Only one class'
                    break
                else:
                    y_pred = model.predict_after_tsfresh(X_test)

                    Classifiers_result[gra][name]['Accuracy'] = accuracy_score(y_test,y_pred)
                    Classifiers_result[gra][name]['Precision'] = precision_score(y_test, y_pred,zero_division=0)
                    Classifiers_result[gra][name]['Recall'] = recall_score(y_test, y_pred,zero_division=0)
                    Classifiers_result[gra][name]['F1'] = f1_score(y_test, y_pred,zero_division=0)
                    Classifiers_result[gra][name]['time'] = model.predict_time_

                with open('models/LathesModel__input_train_{}__fold_{}__gra_{}__PCs_{}__{}.pkl'.format(input_id, it, 
                                                                  model.granularity_,
                                                                  model.N_PCs_,
                                                                  name.replace(' ', '_')), 'wb') as f:
                    pickle.dump(model,f)
                
                

        with open('Classification/Classifiers_result__{}__{}__.pkl'.format(input_id, it), 'wb') as f:
            pickle.dump(Classifiers_result, f)

        model.reset()

        