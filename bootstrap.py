import pandas as pd
import numpy as np
import datetime

signals = pd.read_csv('XBTUSD_hist_2015_11_11_2020_5_20_noML_preproc.csv')
signals['Date'] = pd.to_datetime(signals['Unnamed: 0'],  unit='ns')
signals.drop(['Unnamed: 0'], inplace=True, axis=1)




t1 = pd.Series(signals['Date'].values, index=signals['Date'].values - pd.Timedelta('14 days'),name='t1')
t1.to_csv('XBTUSD_t1')
def getIndMatrix(barIx,t1):
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0,t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1,i]=1
    return indM
indM= getIndMatrix(signals['Date'],t1)
indM.to_csv('XBTUSD_indM')
def getAvgUniquness(indM):
    c = indM.sum(axis=1)
    u = indM.div(c,axis=0)
    avgU = u[u>0].mean()
    return avgU
    
def seqBootstrap(indM, sLength=None):
    if sLength is None: sLength=indM.shape[0]
    phi=[]
    
    while len(phi)<sLength:
        print(len(phi),'of',sLength)
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi+[i]]
            avgU.loc[i]=getAvgUniquness(indM_).iloc[-1]
        prob = avgU/avgU.sum()
        phi+=[np.random.choice(indM.columns,p=prob)]
        
    return phi
phi = seqBootstrap(indM)
signals=signals.iloc[phi]
signals.reset_index(drop=True,inplace=True)
signals.to_csv('XBTUSD_hist_2015_11_11_2020_5_20_noML_bootstrap.csv')
