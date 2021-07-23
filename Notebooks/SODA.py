import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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
