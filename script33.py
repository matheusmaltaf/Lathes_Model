from lathes_model_multiclass import LathesModel
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

from sklearn.metrics import balanced_accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

input_list = np.arange(33)
TRAIN_PATH = 'Input/train_divisions/'
TEST_PATH = 'Input/test_divisions/division__%i__%i__.pkl'

gra_list = [1,2,3,4,5,6,7,8,9,10]

select_list = ['intersection', 'union']

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

folds = 2

for input_id in input_list:
    with (open(TRAIN_PATH %input_id, "rb")) as f:
        train_data = pickle.load(f)
        L1, L2, W = train_data.shape
        train_data = train_data.reshape(L1*L2, W)
        X_train = train_data[:,:-1]
        y_train = train_data[:,-1]
        del train_data
    
    with (open(TEST_PATH (%input_id, %input_id), "rb")) as f:
        test_data = pickle.load(f)
        L1, L2, W = test_data.shape
        test_data = test_data.reshape(L1*L2, W)
        X_test = test_data[:,:-1]
        y_test = test_data[:,-1]
        y_test = y_test[::L2]
        del test_data

    Classifiers_result = {}

    for st in select_list:
        model = LathesModel(N_PCs=4, n_jobs=0)
        for gra in gra_list:
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
                                        'time':0}

                params = {'granularity': gra,
                            'clf': clf,
                            'selection_type':st}

                model.change_hyperparams(params)

                model.fit_after_tsfresh(X_train, y_train)

                if model.one_class_:
                    Classifiers_result[gra]= 'Error: Only one class'
                    break
                else:
                    y_pred = model.predict_after_tsfresh(X_test)
                    acc = balanced_accuracy_score(y_test,y_pred)

                    Classifiers_result[gra][name]['Accuracy'] = acc
                    Classifiers_result[gra][name]['time'] = model.predict_time_

                    f = plt.figure(figsize=(10,7))
                    ax = f.subplots(1,1)
                    plot_confusion_matrix(model.clf, model.X_test_projected_, y_test, 
                        cmap='GnBu', normalize='true', ax=ax)
                    plt.title('{} - {:.2f}%'.format(name, acc))
                    plt.savefig('Figures/Matrix__{}__gra_{}__PCs_{}__{}__{}__.png'.format(input_id, 
                                                                    model.granularity_,
                                                                    model.N_PCs_,
                                                                    name.replace(' ', '_'),
                                                                    st))
                    plt.close()

                    

        with open('Classification/Classifiers_result__{}__{}__.pkl'.format(input_id, st), 'wb') as f:
            pickle.dump(Classifiers_result, f)

        model.reset()

        