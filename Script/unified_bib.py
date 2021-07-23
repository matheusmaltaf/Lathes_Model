import numpy as np
import pandas as pd
import threading
import time
import pickle
import tsfresh
from psutil import cpu_percent
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy as sp
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler,normalize
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import recall_score, f1_score, precision_score
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction import EfficientFCParameters
import os
import glob
from tsfresh.feature_extraction import extract_features
from datetime import datetime

    
def format_func(value, tick_number): #Plot Formater
    # find number of multiples of pi/2
    N = int(value)
    if N == 0:
        return "X1"
    elif N == 50:
        return "X50"
    elif N == 100:
        return "X100"
    elif N == 150:
        return "X150"
    elif N == 200:
        return "X200"
    elif N == 250:
        return "X250"
    elif N == 300:
        return "X300"
    elif N == 350:
        return "X350"
    elif N == 400:
        return "X400"
    elif N == 450:
        return "X450"
    elif N == 500:
        return "X500"
    elif N == 550:
        return "X550"
    elif N == 600:
        return "X600"
    elif N == 650:
        return "X650"
    elif N == 700:
        return "X700"
    elif N == 750:
        return "X750"
    elif N == 800:
        return "X800"
    elif N == 850:
        return "X850"

def tsfresh_chucksize(full_data,input_id):
    # Loading the required input 
    
    L, W = full_data.shape

    data = full_data[:,2:-1]
    info = full_data[:,0:2]

    # Normalizing
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    n_measures = int(max(info[:,1]))

    target = full_data[::n_measures,-1]

    u, idx = np.unique(info[:,0], return_index=True)

    df = pd.DataFrame(np.concatenate((info,data), axis=1), columns= ['id','time'] + 
                        ['Sensor_' + str(x) for x in range(1,W-2)])
    
    with open('Kernel/valid_features_dict__' + input_id + '.pkl', 'rb') as f:
        kind_to_fc_parameters = pickle.load(f)
    
    columns = []
    
    for i,x in enumerate(kind_to_fc_parameters):
        aux = pd.DataFrame(np.hstack((df.loc[:,:'time'].values,
                            df.loc[:,x].values.reshape((-1,1)))),
                            columns=['id','time',x])
        
        aux2 = tsfresh.extract_features(aux, column_id="id", column_sort="time",
                                        default_fc_parameters=kind_to_fc_parameters[x],
                                        #chunksize=3*24000, 
                                        n_jobs=0,
                                        disable_progressbar=False)
        for j in range(len(aux2.columns.tolist())):columns.append(aux2.columns.tolist()[j])

        if i == 0:
            extracted_features = np.array(aux2.values)
        else:
            extracted_features = np.hstack((extracted_features,aux2.values))

    final_features = impute(pd.DataFrame(extracted_features,columns=columns))

    relevance_table = calculate_relevance_table(final_features, target)

    relevant_features = relevance_table[relevance_table.relevant].feature

    filtered_features = final_features.loc[:, relevant_features]
    
    filtered_features.sort_index(inplace = True)
    
    with open('Kernel/final_target_' + input_id + '.pkl', 'wb') as f:
        pickle.dump(target, f)

    # Extracting the selected features dictionary from pandas data frame

    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(filtered_features)

    # Saving dictionary for the on-line phase
    
    with open('Kernel/kind_to_fc_parameters.pkl', 'wb') as f:
        pickle.dump(kind_to_fc_parameters, f)
    
    with open('Kernel/columns.pkl', 'wb') as f:
        pickle.dump(filtered_features.columns.to_list(), f)
        
    Output = {'FeaturesFiltered': filtered_features,
              'FinalTarget': target,
              'RelevanceTable': relevance_table,
              'ID': int(input_id)}
    
    return Output

def tsfresh_chucksize_test(input_id):
    # Loading the required input 
    
    full_data = np.genfromtxt('../Input/Input_' + input_id + '.csv',
                                delimiter=',')
    
    full_data = np.delete(full_data,len(full_data[0])-1,1)

    L, W = full_data.shape

    data = full_data[:,2:-1]
    info = full_data[:,0:2]
    
    # Normalizing
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    n_measures = int(max(info[:,1]))

    target = full_data[::n_measures,-1]

    u, idx = np.unique(info[:,0], return_index=True)

    df = pd.DataFrame(np.concatenate((info,data), axis=1), columns= ['id','time'] + 
                        ['Sensor_' + str(x) for x in range(1,W-2)])
    
    extracted_features = tsfresh.extract_features(df, column_id="id", column_sort="time", n_jobs=0)
    
    return extracted_features

def tsfresh_NaN_filter(input_id,fft=False):
    """
    Given an input_id, this function
    withdraw all NaN features from the 
    TSFRESH extraction; 

    Inputs: 
        -input_id: str() -> the given id
        -fft: True or False -> filter fft features
    
    Outputs:
        - Saves via picklen in ./Kernel/ 
        an extraction dictonary without 
        features that generates NaN
    """

    df = tsfresh_chucksize_test(input_id)
    features = df.columns
    nan_columns = []
    for col in features:
        data = df.loc[:,col].values
        nan_test = np.isnan(data)
        aux  = col.split('__')[1].split('_')[0]
        if aux == 'fft' and fft == True:
            nan_columns.append(col)
        
        elif any(nan == True for nan in nan_test):
            nan_columns.append(col)

    print('Percentage of invalid features: ', len(nan_columns)*100/len(features))

    valid_features = []

    for i in range(len(features)):
        if features[i] not in nan_columns:
            valid_features.append(features[i])
            
    print('Percentage of valid features: ', len(valid_features)*100/len(features))

    valid_features_dict = from_columns(valid_features)

    with open('Kernel/valid_features_dict__' + input_id + '.pkl', 'wb') as f:
            pickle.dump(valid_features_dict, f)

    return

def dynamic_tsfresh (total_data, mode='prototype'):
    ''' Function for ONLINE mode
    This function read the data from the acquisition module and executes a 
    dynamic and lighter version of TSFRESH.
    
    Parameters:
    ------
    input_id : int 
        identifier of the seed dataset
    
    extracted_names: list
    
    Returns: 
    -------
    dataframe #########################################################
        

    '''
        

    data = total_data[:,2:-1]
    info = total_data[:,0:2]
        
    # Normalizing
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    total_data = np.concatenate((info,data), axis=1)
      
    # ----------------------------------------------------------------- # 
    df = pd.DataFrame(total_data, columns= ['id','time'] + 
                        ['Sensor_' + str(x) for x in range(1,(total_data.shape[1]-1))])
    
    # Loading feature dictionary
    with open('Kernel/kind_to_fc_parameters.pkl', 'rb') as f:
        kind_to_fc_parameters = pickle.load(f)
    
    # Loading column names

    with open('Kernel/columns.pkl', 'rb') as f:
        original_columns = pickle.load(f)
    
    columns = []
    

    for i,x in enumerate(kind_to_fc_parameters):
        aux = pd.DataFrame(np.hstack((df.loc[:,:'time'].values,
                            df.loc[:,x].values.reshape((-1,1)))),
                            columns=['id','time',x])
        
        aux2 = tsfresh.extract_features(aux, column_id="id", column_sort="time",
                                        default_fc_parameters=kind_to_fc_parameters[x],#chunksize=24000, 
                                        n_jobs=0
                                        #disable_progressbar=True
                                        )
        for j in range(len(aux2.columns.tolist())):columns.append(aux2.columns.tolist()[j])

        if i == 0:
            extracted_features = np.array(aux2.values)
        else:
            extracted_features = np.hstack((extracted_features,aux2.values))

    final_features = pd.DataFrame(extracted_features,columns=columns)
    final_features = final_features[original_columns]

    return impute(final_features)

def PCA_calc (SelectedFeatures,N_PCs,Chose = 'Analytics',it=0):
    ''' Function to project and execute a Principal Components Analysis
    
    Parameters:
    ------
    SelectedFeatures : dictionary, with the following items
        'FeaturesFiltered': pd.DataFrame
            contain the output data of TSFRESH, i.e., the dataset with features selected by the hypothesis test
        'FinalTarget': np.array
            targets of the entire dataset
        'ID': int
            identifier for the dataset
    
    N_PCs: int
        number of Principal Components to mantain
    
    Chose: str
        type of analysis, can be ['Test', 'Calc', 'Specific', 'Analytics'] 
        (default is 'Analytics')
    
    Returns: 
    -------
    dictionary, with the following items
        'ReducedFeatures': np.array
            contain the output data of PCA, i.e., the dataset with Principal Componentes projected by PCA
        'ID': int
            identifier for the dataset
    
    '''
    selected_features = SelectedFeatures['FeaturesFiltered']
    input_id = SelectedFeatures['ID']

    pca_scaler = StandardScaler()
    scaled_features = pca_scaler.fit_transform(selected_features)

    with open('Kernel/pca_scaler.pkl', 'wb') as f:
        pickle.dump(pca_scaler, f)

    pca = PCA(n_components=N_PCs)
    
    reduced_features = pca.fit_transform(scaled_features)

    variation_kept = pca.explained_variance_ratio_*100

    print(30*'-')
    print(selected_features.shape)
    print(30*'-')
    print(reduced_features.shape)
    print(30*'-')
    print('Percentage of Variation held: {:.2f}%'.format(variation_kept.sum()))

    eigen_matrix = abs(np.array(pca.components_))

    for i in range (eigen_matrix.shape[0]):
        LineSum = sum(eigen_matrix[i,:])
        for j in range (eigen_matrix.shape[1]):
            eigen_matrix[i,j] = ((eigen_matrix[i,j]*100)/LineSum)

    # Weighted Contribution for each feature
    weighted_contribution = (eigen_matrix.T * variation_kept.T).sum(1)/variation_kept.sum()

    selected_columns = selected_features.columns
    df_weighted_contribution = pd.DataFrame(weighted_contribution.reshape(1,-1), columns=selected_columns)                
    df_weighted_contribution = df_weighted_contribution.sort_values(by=0, axis=1, ascending=False)

    #Creating Separated dictionaries for Sensors and Features Contribution 
    sensors_names = [None] * int(df_weighted_contribution.shape[1])
    features_names = [None] * int(df_weighted_contribution.shape[1])
    general_features = [None] * int(df_weighted_contribution.shape[1])

    c = '__'
    for i, names in zip(range (df_weighted_contribution.shape[1]), df_weighted_contribution.columns):
        words = names.split(c)
        sensors_names[i] = words[0]
        general_features[i]= words[1]
        features_names[i] = c.join(words[1:])
                    
    unique_sensors_names = np.unique(sensors_names).tolist()
    unique_general_features = np.unique(general_features).tolist()
    unique_features_names = np.unique(features_names).tolist()

    sensors_contribution = dict.fromkeys(unique_sensors_names, 0)
    general_features_contribution = dict.fromkeys(unique_general_features, 0)
    features_contribution = dict.fromkeys(unique_features_names, 0)      

    #Creating dictionaries from Data Frame orientation
    weighted_contribution = {}
    for col in df_weighted_contribution.columns:
        parts = col.split(c)
                    
        kind = parts[0]
        feature = c.join(parts[1:])
        feature_name = parts[1]
                    
        if kind not in weighted_contribution:
            weighted_contribution[kind] = {}
                    
        sensors_contribution[kind] += df_weighted_contribution.loc[0,col]
        general_features_contribution[feature_name] += df_weighted_contribution.loc[0,col]
        features_contribution[feature] += df_weighted_contribution.loc[0,col]
        weighted_contribution[kind][feature] = df_weighted_contribution.loc[0,col]


    fig = plt.figure(figsize=[16,8])
    fig.suptitle('Percentage of Variance Held by PCs', fontsize=22)
    ax = fig.subplots(1,1)
    ax.bar(x=['PC' + str(x) for x in range(1,(N_PCs+1))],height=variation_kept[0:N_PCs])
    ax.set_ylabel('Percentage of Variance Held',fontsize=27)
    ax.set_xlabel('Principal Components',fontsize=20)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.grid()
    plt.show()
    fig.savefig('PCA_Analyses/Figures/Percentage_of_Variation_held_{}.png'.format(input_id), bbox_inches='tight')

    fig = plt.figure(figsize=[16,8])
    fig.suptitle('Sensors Weighted Contribution Percentage', fontsize=22)
    ax = fig.subplots(1,1)
    s = sensors_contribution
    ax.bar(*zip(*s.items()))
    plt.ylabel('Relevance Percentage',fontsize=27)
    plt.xlabel('Sensors',fontsize=20)
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.show()
    fig.savefig('PCA_Analyses/Figures/Sensors_weighted_contribution_{}.png'.format(input_id), bbox_inches='tight')

    fig = plt.figure(figsize=[16,8])

    fig.suptitle('Features Weighted Contribution Percentage', fontsize=16)
    ax = fig.subplots(1,1)
    s = dict(sorted(features_contribution.items(), 
                            key=lambda item: item[1], reverse=True))
    ax.bar(np.arange(len(s)), s.values())
    plt.ylabel('Relevance Percentage',fontsize=20)
    plt.xlabel('Features',fontsize=20)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=18)
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
    plt.grid()
    plt.show()
    fig.savefig('PCA_Analyses/Figures/Features_weighted_contribution_{}.png'.format(input_id), bbox_inches='tight')

    fig = plt.figure(figsize=[16,8])
    fig.suptitle('Best Features Weighted Contribution Percentage', fontsize=16)
    ax = fig.subplots(1,1)
    s = dict(sorted(features_contribution.items(), 
                            key=lambda item: item[1], reverse=True))
    s_20 = list(s.values())[0:20]

    ax.bar(x=['X' + str(x) for x in range(1,(20+1))],height=s_20)
    plt.ylabel('Relevance Percentage',fontsize=20)
    plt.xlabel('Features',fontsize=20)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=18)
    ax.grid()
    ax.set_ylim([s_20[-1]-0.05,s_20[0]+0.05])
    plt.show()
    fig.savefig('PCA_Analyses/Figures/20th_Best_Features_weighted_contribution_{}.png'.format(input_id), bbox_inches='tight')



    Output = {'ReducedFeatures': reduced_features,
                'ID': input_id} 

    return Output

def PCA_projection (features,N_PCs):
    ''' Function for ONLINE mode
    This function project the data into a trained PCA.
    
    Parameters:
    ------
    features: dataframe 
        #############################################################
    
    Returns: 
    -------
    dataframe
        contain the output data of PCA, i.e., the dataset with Principal Componentes projected by PCA
    
    '''
    loaded_scaler = pickle.load(open('Kernel/pca_scaler.sav', 'rb'))
    features_padronizadas = loaded_scaler.transform(features)
    #centralizar os dados e colocá-los com desvioPadrão=1
    #scaler = StandardScaler().fit(features)
    #features_padronizadas = scaler.transform(features)

    #pca= PCA(n_components = N_PCs)
    #pca.fit(features_padronizadas)
    
    pca = pickle.load(open('Kernel/pca.sav', 'rb'))
    features_reduzidas = pca.transform(features_padronizadas)

    #variacao_percentual_pca = np.round(pca.explained_variance_ratio_ * 100, decimals = 2)

    #print('Variation maintained: %.2f' % variacao_percentual_pca.sum())
    #print('                  ')


    #features_reduzidas = pca.transform(features_padronizadas)

    """
    # load the model from disk
    loaded_pca = pickle.load(open('Kernel/pca.sav', 'rb'))
    
    scaler = StandardScaler().fit(features)
    features_padronizadas = scaler.transform(features)

    features_padronizadas = scaler.transform(features)
    features_reduzidas = loaded_pca.transform(features_padronizadas)
    """
    
    return features_reduzidas

#from numba import njit

def grid_set(data, N):
    '''
    # Stage 1: Preparation

    # --> grid_trad
    # grid_trad é o valor medio da distancia euclidiana entre todo par de data samples dividido pela granularidade


    # --> grid_angl
    # grid_angl é o valor medio da distancia cosseno entre todo par de data samples dividido pela granularidade
    '''
    _ , W = data.shape
    AvD1 = data.mean(0)
    X1 = np.mean(np.sum(np.power(data,2),axis=1))
    grid_trad = np.sqrt(2*(X1 - np.sum(AvD1*AvD1)))/N
    Xnorm = np.sqrt(np.sum(np.power(data,2),axis=1))
    new_data = data.copy()
    for i in range(W):
        new_data[:,i] = new_data[:,i] / Xnorm
    seq = np.argwhere(np.isnan(new_data))
    if tuple(seq[::]): new_data[tuple(seq[::])] = 1
    AvD2 = new_data.mean(0)
    grid_angl = np.sqrt(1-np.sum(AvD2*AvD2))/N
    return X1, AvD1, AvD2, grid_trad, grid_angl

def pi_calculator(Uniquesample, mode):
    '''
    # Calculo da Proximidade Cumulativa na versão recursiva
    # Seção número 2.2.i do SODA
    '''
    UN, W = Uniquesample.shape
    if mode == 'euclidean':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(np.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(np.power(AA1,2))
        aux = []
        for i in range(UN): aux.append(AA1)
        aux2 = [Uniquesample[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.power(aux2,2),axis=1)+DT1

    if mode == 'cosine':
        Xnorm = np.matrix(np.sqrt(np.sum(np.power(Uniquesample,2),axis=1))).T
        aux2 = Xnorm
        for i in range(W-1):
            aux2 = np.insert(aux2,0,Xnorm.T,axis=1)
        Uniquesample1 = Uniquesample / aux2
        AA2 = np.mean(Uniquesample1,0)
        X2 = 1
        DT2 = X2 - np.sum(np.power(AA2,2))
        aux = []
        for i in range(UN): aux.append(AA2)
        aux2 = [Uniquesample1[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.sum(np.power(aux2,2),axis=1),axis=1)+DT2
        
    return uspi

def Globaldensity_Calculator(Uniquesample, distancetype):
    '''
    # Calculo da Densidade Global
    #
    # Além de calcular a densidade global também utiliza o Gaussian KDE para fazer uma aproximação da distribuição dos dados 
    # de treino, em seguida calcula o likelihood dos dados de teste de acordo com o Gaussian KDE construido.
    #
    # Retorna:
    # GD - Densidade Global
    #       Soma das densidades globais para as duas componentes de distance(Euclidiana e Cosseno) multiplicada pelo likelihood
    # Density_1 - Densidade Euclidiana * Likelihood da componente de Densidade Euclidiana
    # Density_2 - Densidade Cosseno * Likelihood da componente de Densidade Cosseno
    # Uniquesample - Amostras organizadas em ordem descendente de Densidade Global
    '''
    uspi1 = pi_calculator(Uniquesample, distancetype)
    
    sum_uspi1 = sum(uspi1)
    Density_1 = uspi1 / sum_uspi1

    uspi2 = pi_calculator(Uniquesample, 'cosine')

    sum_uspi2 = sum(uspi2)
    Density_2 = uspi2 / sum_uspi2

    GD = (Density_2+Density_1)
    index = GD.argsort()[::-1]
    GD = GD[index]
    Uniquesample = Uniquesample[index]


    return GD, Density_1, Density_2, Uniquesample

#@njit(fastmath = True)
def hand_dist(XA,XB):   
    '''
    # Calculo das distancias Euclidiana e Cosseno entre uma amostra (XA) e um conjunto de amostras (XB)
    #
    # A ideia foi fazer da forma mais otimizada que encontrei, sem o uso de listas com tamanho dinamico
    # utilizando um np.array de tamanho fixo, e também utilizando operações matematicas puras do python,
    # sem funções externas.
    '''
    L, W = XB.shape
    distance = np.zeros((L,2))
    
    for i in range(L):
        aux = 0 # Euclidean
        dot = 0 # Cosine
        denom_a = 0 # Cosine
        denom_b = 0 # Cosine
        for j in range(W):
            aux += ((XA[0,j]-XB[i,j])**2) # Euclidean
            dot += (XA[0,j]*XB[i,j]) # Cosine
            denom_a += (XA[0,j] * XA[0,j]) # Cosine
            denom_b += (XB[i,j] * XB[i,j]) # Cosine

        distance[i,0] = aux**.5
        distance[i,1] = ((1 - ((dot / ((denom_a ** 0.5) * (denom_b ** 0.5)))))**2)**.25
    
    return distance
        
#@njit
def chessboard_division_njit(Uniquesample, MMtypicality, grid_trad, grid_angl, distancetype):
    '''
    # Stage 2: DA Plane Projection
    '''
    L, WW = Uniquesample.shape
    W = 1
    
    contador = 0
    BOX = np.zeros((L,WW))
    BOX_miu = np.zeros((L,WW))
    BOX_S = np.zeros(L)
    BOX_X = np.zeros(L)
    BOXMT = np.zeros(L)
    NB = W
    
    BOX[contador,:] = Uniquesample[0,:]
    BOX_miu[contador,:] = Uniquesample[0,:]
    BOX_S[contador] = 1
    BOX_X[contador] = np.sum(Uniquesample[0]**2)
    BOXMT[contador] = MMtypicality[0]
    contador += 1
                   
    for i in range(W,L):
        
        distance = hand_dist(Uniquesample[i].reshape(1,-1),BOX_miu[:contador,:])
        
        SQ = []
        # Condition 1
        # Seção 3.2 do artigo SODA
        # Associar um data sample a um ou mais DA planes
        for j,d in enumerate(distance):
            if d[0] < grid_trad and d[1] < grid_angl:
                SQ.append(j)
        COUNT = len(SQ)

        if COUNT == 0:
            BOX[contador,:] = Uniquesample[i]
            BOX_miu[contador,:] = Uniquesample[i] # Eq. 22b
            BOX_S[contador] = 1 # Eq. 22c
            BOX_X[contador] = np.sum(Uniquesample[i]**2)
            BOXMT[contador] = MMtypicality[i] # Eq. 22d
            NB = NB + 1 # Eq. 22a
            contador += 1

        if COUNT >= 1:
            # Se dois ou mais DA planes satisfazem a condição 1 vale o mais proximo 
            # Eq. 20
            DIS = [distance[S,0]/grid_trad + distance[S,1]/grid_angl for S in SQ] 
            b = 0
            mini = DIS[0]
            for ii in range(1,len(DIS)):
                if DIS[ii] < mini:
                    mini = DIS[ii]
                    b = ii

            BOX_S[SQ[b]] = BOX_S[SQ[b]] + 1 #Eq. 21b
            BOX_miu[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_miu[SQ[b]] + Uniquesample[i]/BOX_S[SQ[b]] # Eq. 21a
            BOX_X[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_X[SQ[b]] + np.sum(Uniquesample[i]**2)/BOX_S[SQ[b]]
            BOXMT[SQ[b]] = BOXMT[SQ[b]] + MMtypicality[i] # Eq. 21c

    BOX_new = BOX[:contador,:]
    BOX_miu_new = BOX_miu[:contador,:]
    BOX_X_new = BOX_X[:contador]
    BOX_S_new = BOX_S[:contador]
    BOXMT_new = BOXMT[:contador]
    return BOX_new, BOX_miu_new, BOX_X_new, BOX_S_new, BOXMT_new, NB

#@njit(fastmath = True)
def ChessBoard_PeakIdentification_njit(BOX_miu,BOXMT,NB,grid_trad,grid_angl, distancetype):
    '''
    # Stage 3: Itendtifying Focal Points
    '''
    Centers = []
    n = 2
    ModeNumber = 0
    L, W = BOX_miu.shape
    
    for i in range(L):
        distance = hand_dist(BOX_miu[i,:].reshape(1,-1),BOX_miu)
        seq = []
        # Condition 2
        for j,(d1,d2) in enumerate(distance):
            if d1 < n*grid_trad and d2 < n*grid_angl:
                seq.append(j)
        Chessblocak_typicality = [BOXMT[j] for j in seq]
        # Condition 3
        # Verificar se o pico local de densidade pertence ao DA plane que está sendo avaliado
        if max(Chessblocak_typicality) == BOXMT[i]:
            Centers.append(BOX_miu[i])
            ModeNumber = ModeNumber + 1
    return Centers, ModeNumber

#@njit(fastmath = True)
def cloud_member_recruitment_njit(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl, distancetype):
    '''
    # Stage 4: Forming Data Clouds
    #
    # Um data sample é associado ao Data Cloud com o ponto focal mais proximo
    #
    '''
    L, W = Uniquesample.shape
    
    B = np.zeros(L)
    for ii in range(L):        
        distance = hand_dist(Uniquesample[ii,:].reshape(1,-1),Center_samples)
        
        dist3 = np.sum(distance, axis=1)
        mini = dist3[0]
        mini_idx = 0
        for jj in range(1, len(dist3)):
            # Condition 4
            if dist3[jj] < mini:
                mini = dist3[jj]
                mini_idx = jj
        B[ii] = mini_idx
    return B

def SelfOrganisedDirectionAwareDataPartitioning(Input):
    data = Input['StaticData']
    L, W = data.shape
    N = Input['GridSize']
    distancetype = Input['DistanceType']

    X1, AvD1, AvD2, grid_trad, grid_angl = grid_set(data,N)
        
    GD, D1, D2, Uniquesample = Globaldensity_Calculator(data, distancetype)

    BOX,BOX_miu,BOX_X,BOX_S,BOXMT,NB = chessboard_division_njit(Uniquesample,GD,grid_trad,grid_angl, distancetype)

    Center,ModeNumber = ChessBoard_PeakIdentification_njit(BOX_miu,BOXMT,NB,grid_trad,grid_angl, distancetype)
     
    IDX = cloud_member_recruitment_njit(ModeNumber,np.array(Center),data,grid_trad,grid_angl, distancetype)
           
        
    Boxparameter = {'BOX': BOX,
                'BOX_miu': BOX_miu,
                'BOX_S': BOX_S,
                'NB': NB,
                'XM': X1,
                'L': L,
                'AvM': AvD1,
                'AvA': AvD2,
                'GridSize': N}

    Output = {'C': Center,
              'IDX': list(IDX.astype(int)+1),
              'SystemParams': Boxparameter,
              'DistanceType': distancetype}
    return Output

def SODA (ReducedFeatures, min_granularity, max_granularity, pace):#SODA
    ''' Start of SODA
    
    Parameters:
    ------
    ReducedFeatures : dictionary, with the following items
        'ReducedFeatures': np.array
            contain the output data of PCA, i.e., the dataset with Principal Componentes projected by PCA
        'ID': int
            identifier for the dataset
    
    min_granularity: float
        first value of granularity for SODA algorithm
    
    max_granularity: float
        final value of granularity for SODA algorithm
        
    pace: float
        increase of granularity for SODA algorithm
    
    Returns: 
    -------
    dictionary, with the following items
        'granularity_i': out['idx']
            Labels of the SODA data clouds
        ...
        'granularity_n': 

        'ID':int
            identifier for the dataset
    '''
    DataSetID = ReducedFeatures['ID']
    data = ReducedFeatures['ReducedFeatures']
    
    #### Looping SODA within the chosen granularities and distances ####
    Output = {}

    for g in np.arange(int(min_granularity), int (max_granularity + pace), pace):

        print('Processing granularity %d'%g)

        Input = {'GridSize':g, 'StaticData':data, 'DistanceType': 'euclidean'}
        
        out = SelfOrganisedDirectionAwareDataPartitioning(Input)

        Output['granularity_' + str(g)] = out['IDX']

    Output['ID'] = DataSetID
    
    return Output

def GroupingAlgorithm (SODA_parameters): #Grouping Algorithm
    print('             ')
    print('Grouping Algorithm Control Output')
    print('----------------------------------')
    
    ####   imput data    ####
    DataSetID = SODA_parameters['ID']
    
    with open('Kernel/final_target_'+str(DataSetID)+'.pkl', 'rb') as f:
        y_original = pickle.load(f)

    Output = {}

    for gra in SODA_parameters:
        if gra  != 'ID':
            SodaOutput = SODA_parameters[gra]
            
            #### Program Matrix's and Variables ####

            define_percent = 50
            n_DA_planes = np.max(SodaOutput)
            Percent = np.zeros((int(n_DA_planes),3))
            n_IDs_per_gp = np.zeros((int(n_DA_planes),2))
            n_tot_Id_per_DA = np.zeros((int(n_DA_planes),1))
            decision = np.zeros(int(n_DA_planes))
            selected_samples = np.zeros(2)
            n_gp0 = 0
            n_gp1 = 0
            k = 0

            #### Definition Percentage Calculation #####

            for i in range(y_original.shape[0]):

                if y_original[i] == 0:
                    n_IDs_per_gp [int(SodaOutput[i]-1),0] += 1 
                else:
                    n_IDs_per_gp [int(SodaOutput[i]-1),1] += 1 

                n_tot_Id_per_DA [int(SodaOutput[i]-1)] += 1 


            for i in range(int(n_DA_planes)):

                Percent[i,0] = (n_IDs_per_gp[i,0] / n_tot_Id_per_DA[i]) * 100
                Percent[i,1] = (n_IDs_per_gp[i,1] / n_tot_Id_per_DA[i]) * 100
            
            #### Using Definition Percentage as Decision Parameter ####

            for i in range(Percent.shape[0]): # pylint: disable=E1136  # pylint/issues/3139

                if (Percent[i,0] > define_percent):
                    n_gp0 += 1
                    decision[i] = 0
                else:
                    n_gp1 += 1
                    decision[i] = 1
                  
            #### Defining labels

            ClassifiersLabel = []

            for i in range (len(SodaOutput)):
                ClassifiersLabel.append(decision[int (SodaOutput[i]-1)])
            
            Output[gra] = ClassifiersLabel

            ### Printig Analitics results
            
            print(gra)
            print('Number of data clouds: %d' % n_DA_planes)
            print('Number of good tools groups: %d' % n_gp0)
            print('Number of worn tools groups: %d' % n_gp1)
            print('Number of samples: %d' % int(len(SodaOutput)))
            print('---------------------------------------------------')

            
            # Saving analysis result
            
            Grouping_Analyse = open("Kernel/Grouping_Analyse__" + str(DataSetID) + '__' + str(gra) + "__.txt","a+")
            Grouping_Analyse.write(gra)
            Grouping_Analyse.write('\nNumber of data clouds: %d\n' % n_DA_planes)
            Grouping_Analyse.write('Number of good tools groups: %d\n' % n_gp0)
            Grouping_Analyse.write('Number of worn tools groups: %d\n' % n_gp1)
            Grouping_Analyse.write('Number of samples: %d\n' % len(SodaOutput))
            Grouping_Analyse.write('---------------------------------------------------')
            
            Grouping_Analyse.close()
    
    return Output

def non_parametric_classification (X_train,X_test,GA_parameters,y_test, delta): #Classifiers

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

    Output = {}
    
    for gra in GA_parameters:

        print('-------------------------------------')
        print(gra)

        Output[gra] = {
        'Nearest Neighbors':{},
        'Linear SVM':{},
        'RBF SVM':{},
        'Gaussian Process':{},
        'Decision Tree':{},
        'Random Forest':{},
        'Neural Net':{},
        'AdaBoost':{},
        'Naive Bayes':{},
        'QDA':{}
        }

        # preprocess dataset, split into training and test part
        Accuracy = np.zeros((len(classifiers) + 1))
        Precision = np.zeros((len(classifiers) + 1))
        Recall = np.zeros((len(classifiers) + 1))
        F1 = np.zeros((len(classifiers) + 1))

        y_train = GA_parameters[gra]
        
        # iterate over classifiers

        for name, clf in zip(Output[gra], classifiers):
            start = datetime.now()
            Output[gra][name] = {
                'metrics':{
                    'Accuracy': 0,
                    'Precision':0,
                    'Recall':0,
                    'F1':0
                },
                'time': 0, 
                'predict':0
            }
            try:
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                Output[gra][name]['predict'] = y_predict = list(clf.predict(X_test))
                Output[gra][name]['metrics']['Accuracy'] = score
                Output[gra][name]['metrics']['Precision'] = precision_score(y_test, y_predict,zero_division=0)
                Output[gra][name]['metrics']['Recall'] = recall_score(y_test, y_predict,zero_division=0)
                Output[gra][name]['metrics']['F1'] = f1_score(y_test, y_predict,zero_division=0)
                Output[gra][name]['time'] = datetime.now() - start + delta
            
            except:
                Output[gra]= 'Error: Only one class'
                break

    return Output