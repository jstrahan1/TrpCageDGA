#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:45:25 2021

@author: jstrahan
"""
import numpy as np
import psutil
import scipy
import os
import gc

"""
Calculates the forward committor from A to B as a function of
    each point.
    Parameters
    ----------
    Basis : list of trajectories
        Basis for the Galerkin expansion. Must be zero in state A and B.
    guess : list of trajectories
        Guess function for the Galerkin expansion. Must be one in state B and zero in state A.
    lag : int
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.  
    
    Returns
    -------
    q : list of trajectories
        List of trajectories containing the values of the forward committor at each point.
"""
def ForwardFK(Basis,guess,lag,i=0,Min=0,Max=1,ReturnTrans=False,Halt=True,InD=None,ReturnCoefs=False,Weights=None):
    B=[]
    r=[]
    w=[]
    d=len(Basis[0][0])
    if Weights is None:
        Weights=list(np.ones_like(np.asarray(guess)))
    for traj,g,ind,ww in zip(Basis,guess,InD,Weights):
        B.append(combine(traj,lag,d,Halt,ind))
        r.append(combine1d(g,lag,d,Halt,ind))
        w.append(combine1d(ww,lag,d,Halt,ind))
    if len(B)>0:
        BasisL=np.concatenate(B)
        r=np.concatenate(r)
        w=np.concatenate(w)
        del B
        gc.collect()
    else:
        BasisL=np.array(B)
        r=np.array(r)
    #h=np.concatenate(h)
    rx=r[0::2]
    ry=r[1::2]
    phix=BasisL[0::2]
    phiy=BasisL[1::2]
    print(phix.shape)
    process = psutil.Process(os.getpid())
    print('Mem Usage= '+str(process.memory_info().rss/(1e9))+' Gb')
    bdga=np.matmul((phix*w[0::2,None]).T,0-(ry-rx)/lag)
    Adga=(phix*w[0::2,None]).T@(phiy-phix)/lag
    if np.linalg.cond(Adga)>1e8:
        Adga+=np.identity(len(Adga[:,0]))*.00
    vdga=scipy.linalg.solve(Adga,bdga)
    print(np.linalg.cond(Adga))
    ans=[]
    if ReturnCoefs:
        return vdga
    for B,g in zip(Basis,guess):
        ans.append(np.clip(np.matmul(B,vdga)+g,Min,Max))
    if ReturnTrans:
        return ans,vdga,np.linalg.cond(Adga)
    else:
        return ans
    
"""
Calculates the backward committor from A to B as a function of
    each point.
    Parameters
    ----------
    Basis : list of trajectories
        Basis for the Galerkin expansion. Must be zero in state A and B.
    guess : list of trajectories
        Guess function for the Galerkin expansion. Must be one in state B and zero in state A.
    lag : int
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.  
    
    Returns
    -------
    q : list of trajectories
        List of trajectories containing the values of the forward committor at each point.
"""
    
def BackwardFK(Basis,guess,lag,COM,InD=None,Halt=True,Min=0,Max=1,ReturnCoefs=False):
    B=[]
    r=[]
    pi=[]
    d=len(Basis[0][0])
    if not Halt:
        InD=np.zeros(len(Basis))
    for traj,g,ind,c in zip(Basis,guess,InD,COM):
        B.append(combine_back(traj,lag,d,Halt,ind))
        r.append(combine1d_back(g,lag,d,Halt,ind))
        pi.append(combine1d_back(c,lag,2,False,1))
    BasisL=np.concatenate(B)
    del B
    gc.collect()
    r=np.concatenate(r)
    pi=np.concatenate(pi)[0::1]/np.sum(np.concatenate(pi)[1::2])
    rx=r[1::2]
    ry=r[0::2]
    phix=BasisL[1::2,:]
    phiy=BasisL[0::2,:]
    N=len(phix)
    #bdga=np.matmul(phiy.T,(0-(rx-ry)))
    #Adga=np.matmul(phiy.T,(phix-phiy))
    bdga=np.matmul(phiy.T,(0-(rx-ry)*pi[1::2]))
    Adga=np.matmul(phiy.T,(phix-phiy)*pi[1::2,None])
    vdga=scipy.linalg.solve(Adga,bdga)
    ans=[]
    
    print(np.linalg.cond(Adga))
    if ReturnCoefs:
        return vdga
    for B,g in zip(Basis,guess):
        ans.append(np.clip(np.matmul(B,vdga)+g,Min,Max))
    return ans
    
"""
Calculates the change of measure to equilibrium as a function of
    each point.
    Parameters
    ----------
    Basis : list of trajectories
        Basis for the Galerkin expansion.  The constant function must be the first.
  lag : int
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.  
    
    Returns
    -------
    q : list of trajectories
        List of trajectories containing the values of the forward committor at each point.
"""
def changeOfMeasure_list_noeig(Basis,lag,skip=0,Weights=None):
    B=[]
    W=[]
    d=len(Basis[0][0])
    if Weights is None:
        Weights=list(np.ones_like(np.asarray(Basis)[:,:,0]))
    for traj,w in zip(Basis,Weights):
        B.append(combine(traj,lag,d,False,None))
        W.append(combine1d(w,lag,d,False,None))
    BasisL=np.concatenate(B)
    W=np.concatenate(W)
    phix=BasisL[0::2,:]
    phiy=BasisL[1::2,:]
    N=len(phix)
    Ct=(phix*W[0::2,None]).T@phiy/N
    C0=(phix*W[0::2,None]).T@phix/N
    vec=np.ones(len(Ct[0]))
    p=np.linalg.solve((Ct-C0).T[1:,1:],-(Ct-C0).T[1:,0])
    vec[1:]=p
    ans=[]
    for B in Basis:
        ans.append(B@vec)
    return ans

"""
Calculates the reactve current on a grid.
    Parameters
    ----------
    fx,fy : list of trajectories
        CV space to compute reactive current on.
    qf,qb : list of trajectories
        Forward and Backward committors.
    COM : list of trajectories
        change of measure to equilibrium.
    InD : list of trajectories
        Indicator function on (AUB)^c.
    xlim,ylim: ndarray
        limits defining the gridding on the x and y CVs.
    lag : int
        Number of timepoints in the future to use for the finite difference in the discrete-time generator.  
    
    Returns
    -------
    JAB : ndarray 
        Vector field of shape (2,len(xlim),len(ylim)) containing the reactive current at each grid cell.
"""
def Flux_Final(fx,fy,qf,qb,COM,InD,xlim,ylim,lag=1):
    N=len(COM)
    lentraj=len(COM[0])
    FXF=[]
    FYF=[]
    Pi=[]
    QFF=[]
    QBF=[]
    FXB=[]
    FYB=[]
    Ind=[]
    QFB=[]
    QBB=[]
    d=5
    for x,y,pi,f,b,c in zip(fx,fy,COM,qf,qb,InD):
        FXB.append(combine1d_back(x,lag,d,True,c))
        FYB.append(combine1d_back(y,lag,d,True,c))
        FXF.append(combine1d(x,lag,d,True,c))
        FYF.append(combine1d(y,lag,d,True,c))
        QFF.append(combine1d(f,lag,d,True,c))
        QFB.append(combine1d_back(f,lag,d,True,c))
        QBF.append(combine1d(b,lag,d,True,c))
        QBB.append(combine1d_back(b,lag,d,True,c))
        Pi.append(combine1d(pi,lag,5,False,1)) 
    fxf=np.concatenate(FXF).flatten()
    fyf=np.concatenate(FYF).flatten()
    fxb=np.concatenate(FXB).flatten()
    fyb=np.concatenate(FYB).flatten()
    COM=np.clip(np.concatenate(Pi),0,1000)
    QFF=np.concatenate(QFF)
    QBF=np.concatenate(QBF)
    QFB=np.concatenate(QFB)
    QBB=np.concatenate(QBB)
    #Ind=np.concatenate(Ind)
    X,Y=np.meshgrid(xlim,ylim)
    print(X.shape)
    pmf=np.zeros((2,len(X)-1,len(X)-1))
    i=0
    for x in range(len(xlim)-1):
        for y in range(len(ylim)-1):
            Indsf=np.where(np.logical_and(np.logical_and(fxf<xlim[x+1],fxf>xlim[x]),np.logical_and(fyf<ylim[y+1],fyf>ylim[y])))[0]
            InThetaf=np.zeros_like(QFF)
            InThetaf[Indsf]=1
            
            Indsb=np.where(np.logical_and(np.logical_and(fxb<xlim[x+1],fxb>xlim[x]),np.logical_and(fyb<ylim[y+1],fyb>ylim[y])))[0]
            InThetab=np.zeros_like(QFB)
            InThetab[Indsb]=1
            
            numx=np.sum(QFF[1::2]*QBF[0::2]*(fxf[1::2]-fxf[0::2])*InThetaf[0::2])
            numx+=np.sum(QFB[0::2]*QBB[1::2]*(fxb[0::2]-fxb[1::2])*InThetab[0::2])
            numy=np.sum(QFF[1::2]*QBF[0::2]*(fyf[1::2]-fyf[0::2])*InThetaf[0::2])
            numy+=np.sum(QFB[0::2]*QBB[1::2]*(fyb[0::2]-fyb[1::2])*InThetab[0::2])
            pmf[0,x,y]=.5*numx/np.sum(COM[0::2])
            pmf[1,x,y]=.5*numy/np.sum(COM[0::2])
            i+=1
    return pmf

def PMF_grid(fx,fy,COM,xlim,ylim,lag):
    N=len(COM)
    FX=[]
    FY=[]
    Pi=[]
    
    for x,y,pi in zip(fx,fy,COM):
       FX.append(combine1d(x,lag,2,False,1))
       FY.append(combine1d(y,lag,2,False,1))
       Pi.append(combine1d(pi,lag,5,False,1)) 
    fx=np.concatenate(FX)[0::2]
    fy=np.concatenate(FY)[0::2]
    COM=np.clip(np.concatenate(Pi)[0::2],0,1000)
    X,Y=np.meshgrid(xlim,ylim)
    print(X.shape)
    pmf=np.zeros((len(X)-1,len(X)-1))
    i=0
    for x in range(len(xlim)-1):
        for y in range(len(ylim)-1):
            inds=np.where(np.logical_and(np.logical_and(fx<xlim[x+1],fx>xlim[x]),np.logical_and(fy<ylim[y+1],fy>ylim[y])))[0]
            pmf[x,y]=np.sum(COM[inds])/np.sum(COM)
            i+=1
    return pmf

def avg_on_pmf_dga(fx,fy,q,COM,xlim,ylim,lag):
    N=len(COM)
    M=len(COM[0])
    Q=[]
    FX=[]
    FY=[]
    Pi=[]
    X,Y=np.meshgrid(xlim,ylim)
    print(X.shape)
    pmf=np.zeros((len(X)-1,len(X)-1))
    for x,y,pi,qd in zip(fx,fy,COM,q):
       FX.append(combine1d(x,lag,2,False,1))
       FY.append(combine1d(y,lag,2,False,1))
       Pi.append(combine1d(pi,lag,5,False,1))
       Q.append(combine1d(qd,lag,5,False,1)) 
    fx=np.concatenate(FX)[0::2]
    fy=np.concatenate(FY)[0::2]
    COM=np.clip(np.concatenate(Pi)[0::2],0,1000)
    q=np.concatenate(Q)[0::2]
    i=0
    for x in range(len(xlim)-1):
        for y in range(len(ylim)-1):
            inds=np.where(np.logical_and(np.logical_and(fx<xlim[x+1],fx>xlim[x]),np.logical_and(fy<ylim[y+1],fy>ylim[y])))[0]
            pmf[x,y]=np.sum(q[inds]*COM[inds])/np.sum(COM[inds])
            i+=1
    return pmf

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def find(a):
    ans=np.where(a)[0]
    if len(ans)==0:
        return -1
    else:
        return ans[0]
    
#Time-Lag a single Trajectory.
def combine(x,lag,d,Halt,InD):
    if Halt==False or lag==1:
        Data=np.zeros((2*(len(x)-lag),d))
        Data[0::2]=x[0:len(x)-lag]
        Data[1::2]=x[lag:len(x)]
    else:
        D=np.copy(rolling_window(InD,lag+1))
        #print(D)
        offset=np.argsort(D,kind='mergesort')[:,0]
        #print(offset)
        #print(np.sum(np.mean(D,axis=1)==1))
        offset[np.mean(D,axis=1)==1]=lag
        temp=np.where(D[:,0]-D[:,1]==-1)[0]
        offset[temp]=np.argsort(D[:,1:],kind='mergesort')[temp,1]+1
        for t in temp:
            if np.mean(D[t,1:])==1:
                offset[t]=lag
        inds=np.arange(0,len(x)-lag)+offset
        Data=np.zeros((2*(len(x)-lag),d))
        Data[0::2]=x[0:len(x)-lag]
        Data[1::2]=x[inds]    
        
    return Data

#Time-Lag a single Trajectory.
def combine_back(x,lag,d,Halt,InD):
    if Halt==False:
        Data=np.zeros((2*(len(x)-lag),d))
        Data[0::2]=x[lag:]
        Data[1::2]=x[0:len(x)-lag]
    else:
        D=rolling_window(InD,lag+1)
        offset=np.argsort(np.flip(D,axis=1),kind='mergesort')[:,0]
        #print(np.sum(np.mean(D,axis=1)==1))
        offset[np.mean(D,axis=1)==1]=lag
        inds=np.arange(lag,len(x))-offset
        Data=np.zeros((2*(len(x)-lag),d))
        Data[0::2]=x[lag:]
        Data[1::2]=x[inds]   
    return Data


def combine1d(x,lag,d,Halt,InD,ReturnOffset=False):
   if Halt==False or lag==1:
        Data=np.zeros((2*(len(x)-lag)))
        Data[0::2]=x[0:len(x)-lag]
        Data[1::2]=x[lag:len(x)]
        return Data
   else:
        D=np.copy(rolling_window(InD,lag+1))
        #print(D)
        offset=np.argsort(D,kind='mergesort')[:,0]
        #print(offset)
        #print(np.sum(np.mean(D,axis=1)==1))
        offset[np.mean(D,axis=1)==1]=lag
        temp=np.where(D[:,0]-D[:,1]==-1)[0]
        offset[temp]=np.argsort(D[:,1:],kind='mergesort')[temp,1]+1
        for t in temp:
            if np.mean(D[t,1:])==1:
                offset[t]=lag
        inds=np.arange(0,len(x)-lag)+offset
        Data=np.zeros((2*(len(x)-lag)))
        Data[0::2]=x[0:len(x)-lag]
        Data[1::2]=x[inds]
        if ReturnOffset:
            return Data,offset
        else:
            return Data
    
#Time-Lag a single Trajectory.
def combine1d_back(x,lag,d,Halt,InD):
    if Halt==False:
        Data=np.zeros((2*(len(x)-lag)))
        Data[0::2]=x[lag:]
        Data[1::2]=x[0:len(x)-lag]
    else:
        D=rolling_window(InD,lag+1)
        offset=np.argsort(np.flip(D,axis=1),kind='mergesort')[:,0]
        #print(np.sum(np.mean(D,axis=1)==1))
        offset[np.mean(D,axis=1)==1]=lag
        inds=np.arange(lag,len(x))-offset
        Data=np.zeros((2*(len(x)-lag)))
        Data[0::2]=x[lag:]
        Data[1::2]=x[inds]   
    return Data



















