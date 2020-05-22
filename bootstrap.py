import pandas as pd
import numpy as np
import datetime

trades_df=pd.read_csv('hist_2015_11_11-2020_5_20_noML.csv')
trades_df['Opened'] = pd.to_datetime(trades_df['Opened'],  unit='ns')
trades_df['Closed'] = pd.to_datetime(trades_df['Closed'],  unit='ns')


total_trades=trades_df.shape[0]
won_trades= trades_df[trades_df['Profit']>0].shape[0]
ratio =won_trades/total_trades
target_ratio=0.4
need = round((target_ratio-ratio)*total_trades)
won_idx=trades_df[trades_df['Profit']>0].index
random_idx=np.random.choice(won_idx, size=need)
won_df = trades_df.iloc[random_idx]
trades_df = pd.concat([trades_df,won_df])
trades_df = trades_df.sample(frac=1).reset_index(drop=True)

won_trades= trades_df[trades_df['Profit']>0].shape[0]
won_idx=trades_df[trades_df['Profit']>0].index
loss_idx = trades_df[trades_df['Profit']<0].index
new_loss_idx=np.random.choice(loss_idx, size=won_trades)
new_idx= np.concatenate([won_idx,new_loss_idx])
np.random.shuffle(new_idx)
trades_df=trades_df.iloc[new_idx]

trades_df.sort_values(by=['Opened'], inplace=True)
trades_df.reset_index(inplace=True,drop=True)

t1 = pd.Series(trades_df['Opened'].values, index=trades_df['Opened'].values - pd.Timedelta('14 days'),name='t1')

def getIndMatrix(barIx,t1):
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0,t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1,i]=1
    return indM
indM= getIndMatrix(trades_df['Opened'],t1)

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
trades_df=trades_df.iloc[phi]
trades_df.reset_index(drop=True,inplace=True)
trades_df.to_csv('hist_2015_11_11-2020_5_20_noML_bootstrap.csv')
