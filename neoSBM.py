"""runNeoSBM.py - neoSBM main module
    Copyright (C) 2016 Leto Peel

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA"""


import numpy as np
import os
from scipy.special import gammaln, betaln
from random import sample
from sys import stdout
from random import sample
from itertools import izip
import time


def xlogy(x, y):
    return x*np.log(y+1e-200)


def invlogit(a, b):
    if np.min([a,b])<-500:
        c=np.max([a,b])
        a-=c
        b-=c
    expa=np.exp(a)
    return expa/(expa+np.exp(b))


class SBMmh(object):
    
    
    def logy(self,y):
        try:
            return self.log_cache[y]
        except:
            ans=np.log(y+1e-200)
            self.log_cache[y]=ans
            return ans
    
    def __init__(self, E, M,K=None,burnin=0,iterations=1e3,thetas=[.5],theta_idx=0,epsilon=1,b=None,c=None,nm=10):
        
        self.e=E   #edgelist
        self.N=len(M)       #number of nodes
        self.I=len(E)          #number of edges
        if K is None:
            self.K=len(np.unique(M)) #number of metadata categories
        else:
            self.K=K
        self.x=M        # metadata
        self.burnin = burnin
        self.maxiterations = iterations
        
        self.nm=nm
        
        self.epsilonB=epsilon*self.K
        
        self.theta_idx=theta_idx
        self.thetas=np.array(thetas)
        self.thetaln=np.array([np.log(theta) for theta in thetas]) #xlogy(1,theta)
        self.minusthetaln=np.array([np.log(1-theta) for theta in thetas]) #xlogy(1,1-theta)
        self.minusthetaln[self.thetas==1] = -1e200
        self.thetaln[self.thetas==0] = -1e200
        
        
        self.minEntropy=-np.inf
        #~ self.max_neoLL=np.inf
        self.neoLL=0
        
        #find min possible LLs using metadata initialisation
        #~ try:
            #~ self.minEntropy=-previous_max['LL']
            #~ self.max_neoLL=previous_max['neoLL']
            #~ self.cmax=previous_max['cmax']
            #~ self.bmax=previous_max['bmax']
            #~ print "Previous LLs", -self.minEntropy, self.max_neoLL
        #~ except KeyError:
        self.initialiseBC(b=np.zeros(self.N),c=M.copy())
        self.minEntropy=self.calcEntropy(0)
        self.max_LL=np.zeros(len(thetas))-self.minEntropy
        self.max_neoLL=np.zeros(len(thetas))-self.minEntropy + self.N*self.minusthetaln
        self.cmax= np.repeat(M[:,np.newaxis],len(thetas),axis=1)#M.copy()
        self.bmax=np.zeros((self.N,len(thetas)))
        #~ print "Metadata LLs", -self.minEntropy, self.max_neoLL
            
        self.temp=1000.
        #~ if np.random.rand()>0.5:
        self.initialiseBC(b=b,c=c)
        self.entropy=self.calcEntropy(0)
        #~ print self.c
        
        self.total_iterations=0
        
        self.log_cache={}
        self.t_change=0
        
    
    
    
    """initialise counts related to b and c, including number of links
    (and possible links) between blocks """
    def initialiseBC(self,c=None,b=None):
        
        #initialise sufficient statistic counts
        if c is None:
            self.c = np.random.randint(0,self.K,size=self.N)
        else:
            self.c = c
        self.edge_c=np.empty(np.shape(self.e))
        for i,e in enumerate(self.e):
            self.edge_c[i,:]=self.c[e]
        self.ccount = np.zeros((self.N,self.K)) #count of node assignments to communities
        self.cnorm = np.zeros(self.N) #count of node assignments to communities
        self.nr=np.zeros(self.K)
        self.drs_sum=np.zeros(self.K)
        self.drs=np.zeros((self.K,self.K))
        self.dis=np.zeros((self.N,self.K))
        
        if b is None:
            self.b=np.random.binomial(1,self.thetas[self.theta_idx],size=self.N)
        else:
            self.b=b
        self.bnorm = np.zeros(self.N) 
        self.edge_to = dict((v,[]) for v in xrange(self.N))
        self.q=self.b.sum()
        
        self.c[self.b==0]=self.x[self.b==0]
        for r in xrange(self.K):
            self.nr[r] = np.sum(self.c==r)
            
            
        for (i,j) in self.e:
            if i!=j:
                self.edge_to[i].append(j)
                self.edge_to[j].append(i)
                
                self.dis[j,self.c[i]]+=1
                self.dis[i,self.c[j]]+=1
                self.drs[self.c[i],self.c[j]]+=1
                self.drs[self.c[j],self.c[i]]+=1
                    
        self.bcount = np.zeros((self.N,2))
        for i in xrange(self.N):
            self.edge_to[i] = np.int32(self.edge_to[i])
            self.drs_sum[self.c[i]]+=1
        
    
    """Calculate entropy (negative log likelihood)
    """
    def calcEntropy(self,t):
        rows,cols=self.drs.nonzero()
        drs=self.drs[rows,cols]
        nrns=self.nr[rows]*self.nr[cols]   
        prs=drs/nrns
        entropy= (-xlogy(drs,prs)-xlogy(nrns-drs,1-prs)).sum()/2.
        return entropy
    
    def calcBlockEntropyDiff(self,r_old,r_i):
        
        drs1=self.drs[[r_old,r_i],:]
        drs2=drs1.sum(0)
        drs2[r_i]+=drs1[:,r_old].sum()
        drs2[r_old]=0
        drs1[1,0]=0
        drs1[[0,1],[r_old,r_i]]/=2.
        drs2[r_i]/=2.
        
        nrns1=np.dot(self.nr[[r_old,r_i],np.newaxis],self.nr[np.newaxis,:])
        nrns2=nrns1.sum(0)
        nrns2[r_i]+=self.nr[r_old] #nrns1[:,r_old].sum()
        nrns2[r_old]=0
        nrns1[1,0]=0
        nrns1[[0,1],[r_old,r_i]]/=2.
        nrns2[r_i]/=2.
        
        cols,=drs2.nonzero()
        drs1=drs1[:,cols]
        nrns1=nrns1[:,cols]
        drs2=drs2[cols]
        nrns2=nrns2[cols]
        
        prs1=drs1/(nrns1+1e-200)
        prs2=drs2/(nrns2+1e-200)
        
        #swap signs here because argsort puts lowest first
        #i.e., delta = new_entropy - old_entropy, 
        #so that lowest delta means highest decrease in entropy
        entropy= - (-xlogy(drs1,prs1)-xlogy(nrns1-drs1,1-prs1)).sum() + (-xlogy(drs2,prs2)-xlogy(nrns2-drs2,1-prs2)).sum() 
        return entropy
    
    
    def updateB(self,i,t):
        b_old = self.b[i]
        c_old = self.c[i]
        x_i = self.x[i]
        
        #sample from a uniform distribution
        b_i = np.random.binomial(1,0.5)
        
        #accept or reject
        if b_i == b_old: # if no change then accept reject has no impact
            if b_i:
                self.updateC(i,t)
        else: 
            
            self.entropy=self.calcEntropy(t)
            logp = -self.entropy #likelihood term
            logp += (self.q)*self.thetaln + (self.N-self.q)*self.minusthetaln #p(b|th)
            self.neoLL=logp[self.theta_idx]
            old_entropy=self.entropy #cache entropy value
            
            if b_i:
                self.updateC(i,t)
            else:
                indices,=self.dis[i,:].nonzero()
                dis=self.dis[i,indices]
                self.nr[c_old]-=1
                self.drs[c_old,indices]-=dis
                self.drs[indices,c_old]-=dis
                self.nr[x_i]+=1
                self.drs[x_i,indices]+=dis
                self.drs[indices,x_i]+=dis
                
                #recalculate entropy - (this is done already by updateC
                self.entropy=self.calcEntropy(t)
            
            qdiff=b_i*2-1
            
            new_logp = -self.entropy #likelihood term
            new_logp += (self.q+qdiff)*self.thetaln + (self.N-self.q-qdiff)*self.minusthetaln #p(b|th)
            
            p_acceptance=min(np.exp((new_logp[self.theta_idx]-logp[self.theta_idx])/self.temp),1)
            #update
            if np.random.binomial(1, p_acceptance):
                self.bacceptance+=1
                self.b[i]=b_i
                
                new_max_idxs = (new_logp > self.max_neoLL).nonzero()[0]
                #~ print new_max_idxs
                self.max_neoLL[new_max_idxs] = new_logp[new_max_idxs]
                self.max_LL[new_max_idxs] = -self.entropy
                self.minEntropy=self.entropy
                self.cmax[:,new_max_idxs]=self.c.copy()[:,np.newaxis]
                self.bmax[:,new_max_idxs]=self.b.copy()[:,np.newaxis]
                
                #~ print self.cmax
                
                #~ if new_logp > self.max_neoLL:
                    #~ self.max_neoLL=new_logp
                    #~ self.minEntropy=self.entropy
                    #~ self.cmax=self.c.copy()
                    #~ self.bmax=self.b.copy()
                
                if b_i:
                    self.q+=1
                    
                else:
                    self.q-=1
                    self.dis[self.edge_to[i],c_old]-=1
                    self.dis[self.edge_to[i],x_i]+=1
                    self.c[i]=x_i
                    
            else:
                if b_i:
                    c_new=self.c[i]
                    indices,=self.dis[i,:].nonzero()
                    dis=self.dis[i,indices]
                    self.nr[c_new]-=1
                    self.drs[c_new,indices]-=dis
                    self.drs[indices,c_new]-=dis
                    self.nr[x_i]+=1
                    self.drs[x_i,indices]+=dis
                    self.drs[indices,x_i]+=dis
                    self.dis[self.edge_to[i],c_new]-=1
                    self.dis[self.edge_to[i],x_i]+=1
                    self.c[i]=x_i
                
                else:
                    self.nr[c_old]+=1
                    self.drs[c_old,indices]+=dis
                    self.drs[indices,c_old]+=dis
                    self.nr[x_i]-=1
                    self.drs[x_i,indices]-=dis
                    self.drs[indices,x_i]-=dis
                self.entropy=old_entropy
        #~ self.entropy = self.calcEntropy()
            #~ print t, self.burnin
        if t > self.burnin:
            self.bcount[i,self.b[i]] += 1
            self.ccount[i,self.c[i]] += 1
            self.cnorm[i] += 1
        
    
    def updateC(self,i,t):
        c_old=self.c[i]
        
        #sample from proposal distribution
        try:
            j=sample(self.edge_to[i],1)
        except ValueError:  #catch singleton nodes
            j=sample(np.arange(self.K),1)
        c_j=self.c[j]
        dt=self.drs[c_j,:].flatten()
        dt_idx,=dt.nonzero()
        dtsum=dt[dt_idx].sum()
        Rt=self.epsilonB/(dtsum + self.epsilonB)
        if np.random.binomial(1,Rt):
            c_i=np.random.randint(self.K)
        else:
            dt[dt_idx]/=dtsum
            c_i=np.random.multinomial(1,dt).argmax()
        
        #accept or reject
        pr_s=0.
        for j in self.edge_to[i]:
            c_j=self.c[j]
            dt=self.drs[c_j,:].flatten()
            dt_idx,=dt.nonzero()
            dtsum=dt[dt_idx].sum()
            Rt=self.epsilonB/(dtsum + self.epsilonB)
            dt[dt_idx]/=dtsum
            pr_s += dt[c_i]*(1-Rt)+Rt/self.K
        try:
            pr_s/=len(self.edge_to[i])
        except ZeroDivisionError:
            pr_s=1
        
        entropy_difference=self.calcEntropy(t)
        
        indices,=self.dis[i,:].nonzero()
        dis=self.dis[i,indices]
        self.nr[c_old]-=1
        self.drs[c_old,indices]-=dis
        self.drs[indices,c_old]-=dis
        self.nr[c_i]+=1
        self.drs[c_i,indices]+=dis
        self.drs[indices,c_i]+=dis
        
        self.entropy=self.calcEntropy(t)
        entropy_difference-=self.entropy
        
        ps_r=0.
        for j in self.edge_to[i]:
            c_j=self.c[j]
            dt=self.drs[c_j,:].flatten()
            dt_idx,=dt.nonzero()
            dtsum=dt[dt_idx].sum()
            Rt=self.epsilonB/(dtsum + self.epsilonB)
            dt[dt_idx]/=dtsum
            ps_r += dt[c_old]*(1-Rt)+Rt/self.K
        try:
            ps_r/=len(self.edge_to[i])
        except ZeroDivisionError:
            ps_r=1
        
        p_acceptance=min(np.exp(entropy_difference)*ps_r/(pr_s+1e-200),1)
        
        
        #update
        if np.random.binomial(1, p_acceptance):
            self.acceptance+=1
            self.dis[self.edge_to[i],c_old]-=1
            self.dis[self.edge_to[i],c_i]+=1
            self.c[i]=c_i
            
        else:
            self.nr[c_old]+=1
            self.drs[c_old,indices]+=dis
            self.drs[indices,c_old]+=dis
            self.nr[c_i]-=1
            self.drs[c_i,indices]-=dis
            self.drs[indices,c_i]-=dis
    
    
    
    def updateBlockC(self,r_old):
        
        #sample from proposal distribution (Random moves)
        #~ s=np.random.multinomial(1, self.drs[r_old,:].flatten()/self.drs[r_old,:].sum()).argmax() #sample neighbour block proportional to edges between blocks
        t=np.random.multinomial(1, self.transition[r_old,:]).argmax() #sample neighbour block proportional to edges between blocks
        
        Rt=self.dr[t] #Rt=self.epsilonB/(self.dr[t] + self.epsilonB)
        if np.random.binomial(1,Rt):
            r_is=np.unique(np.random.randint(self.K,size=self.nm))
        else:
            r_is,=np.random.multinomial(self.nm, self.transition[t,:]).nonzero()
        
        for r_i in r_is:
            #don't bother repeating moves...
            if (r_i,r_old) not in self.moves:
                
                self.entropydiffs.append(self.calcBlockEntropyDiff(r_old,r_i))
                self.moves.append((r_old,r_i))
        
        
        
    
    def merge(self,K):
        tstart=time.time()
        
        t=0
        self.entropydiffs=[]
        self.moves=[]
        self.dr=self.drs.sum(1)
        self.transition=self.drs/(self.dr[:,np.newaxis]+1e-200)
        #cache Rt=self.epsilonB/(self.dr[t] + self.epsilonB)
        self.dr=self.epsilonB/(self.dr + self.epsilonB)
        
        for r in xrange(self.K):
            self.updateBlockC(r)
        
        moveGenerator = (self.moves[m] for m in np.argsort(self.entropydiffs))
        moveMap=np.arange(self.K)
        
        while self.K > K:
            r,s=moveGenerator.next()
            moveMap[moveMap==r]=s
            self.K=len(np.unique(moveMap))
            
        #next ensure that class labels are mapped to [1,K]
        reduceMap= dict((m,r) for m,r in zip(np.unique(moveMap),np.arange(K)))
        
        for i in xrange(self.N):
            self.c[i]=reduceMap[moveMap[self.c[i]]]
        
    
    
    def infer(self):
        self.t=0
        self.temp_init=1000. ** (1-float(self.theta_idx)/len(self.thetas))
        self.temp=self.temp_init
        print "temp init:",self.temp_init
        LLs=[]
        self.bestLL=-np.inf
        self.bestLL2=-np.inf
        #~ self.minEntropy=np.inf
        if self.maxiterations == 0:
            for v in xrange(self.N):
                self.ccount[v,self.c[v]] += 1
        else:
            #~ print "Running M-H sampler:"
            #~ self.recount("start")
            while self.t < self.maxiterations+1:
                if self.t>self.burnin:
                    self.total_iterations+=1
                self.acceptance=0
                self.bacceptance=0
                #~ tstart=time.time()
                for i in sample(xrange(self.N),self.N):
                    #~ print "B",i
                    #~ self.updateC(i,t)
                    self.updateB(i,self.t)
                #~ stdout.write(" %i%% Complete %i (%i) Accepted (q=%i), %f, Entropy= %f (%f)\r" %((self.t+1)*100/self.maxiterations,self.acceptance,self.bacceptance,self.q,self.temp,self.entropy,self.neoLL))
                stdout.write(" %i%% Complete %i (%i) Accepted (q=%i), %f, Entropy= %f\r" %((self.t+1)*100/self.maxiterations,self.acceptance,self.bacceptance,self.q,self.temp,self.entropy))
                stdout.flush()
                self.t += 1
                self.temp=1.+ 0.5*(self.temp_init - 1)*(1+np.cos(2*self.t*np.pi/self.maxiterations))*(self.t<self.maxiterations/2.)


class DCSBMmh(SBMmh):
    """Calculate entropy (negative log likelihood)
    """
    def calcEntropy(self,t):
        rows,cols=self.drs.nonzero()
        drs=self.drs[rows,cols]
        dr=np.sum(self.drs,0)
        drds=dr[rows]*dr[cols]   
        prs=drs/drds
        entropy= (-xlogy(drs,prs)).sum()/2.
        if entropy<self.minEntropy:
            self.minEntropy=entropy
            self.t_change=t
            self.t=0
        return entropy
    
    def calcBlockEntropyDiff(self,r_old,r_i):
        
        drs1=self.drs[[r_old,r_i],:]
        drs2=drs1.sum(0)
        drs2[r_i]+=drs1[:,r_old].sum()
        drs2[r_old]=0
        drs1[1,0]=0
        drs1[[0,1],[r_old,r_i]]/=2.
        drs2[r_i]/=2.
        
        dr=self.drs.sum(1)
        #~ dr2=drs2.sum()
        drds1=np.dot(dr[[r_old,r_i],np.newaxis],dr[np.newaxis,:])
        drds2=drds1.sum(0)
        drds2[r_i]+=dr[r_old] #nrns1[:,r_old].sum()
        drds2[r_old]=0
        drds1[1,0]=0
        drds1[[0,1],[r_old,r_i]]/=2.
        drds2[r_i]/=2.
        
        #~ nrns1=np.dot(self.nr[[r_old,r_i],np.newaxis],self.nr[np.newaxis,:])
        #~ nrns2=nrns1.sum(0)
        #~ nrns2[r_i]+=self.nr[r_old] #nrns1[:,r_old].sum()
        #~ nrns2[r_old]=0
        #~ nrns1[1,0]=0
        #~ nrns1[[0,1],[r_old,r_i]]/=2.
        #~ nrns2[r_i]/=2.
        
        cols,=drs2.nonzero()
        drs1=drs1[:,cols]
        drds1=drds1[:,cols]
        drs2=drs2[cols]
        drds2=drds2[cols]
        
        prs1=drs1/(drds1+1e-200)
        prs2=drs2/(drds2+1e-200)
        
        #swap signs here because argsort puts lowest first
        #i.e., delta = new_entropy - old_entropy, 
        #so that lowest delta means highest decrease in entropy
        entropy= - (-xlogy(drs1,prs1)).sum() + (-xlogy(drs2,prs2)).sum() 
        return entropy


def timer():
    import timeit
    LLs=[]
    with open('synth.txt') as f:
        E=np.int32([row.strip().split() for row in f.readlines()])
    #~ for th in np.arange(0,1.01,0.04):
    #~ M=np.arange(1000)
    M=np.ones(1000)
    M[0]=0
    M[2]=2
    iterations=1000
    burnin=iterations/2
    sbm=SBMmh(E,M,theta=1,burnin=burnin,iterations=iterations)
    sbm.minEntropy=np.inf
    print timeit.timeit(sbm.calcEntropy,number=10000)
    #~ print timeit.timeit(sbm.calcEntropy_,number=1000)
    #~ print timeit.timeit(sbm.calcEntropyDiff,number=1000)


def greedyAgglom(E,M,targetK,initK,theta=1,sigmaK=1.2,sbmModel=SBMmh):
    
    #~ iterations=100
    iterations=0
    
    N=len(M)
    newK=N
    #~ newK=initK
    #~ c=None
    b=None
    c=np.arange(N)
    #~ sbm=SBMmh(E,M,K=int(newK),theta=theta,burnin=iterations/2,iterations=iterations,c=c,b=b)
    sbm=sbmModel(E,M,K=int(newK),theta=1,burnin=iterations/2,iterations=iterations,c=c,b=b)
    newK/=sigmaK
    sbm.merge(int(newK))
    c=sbm.c.copy()
    b=sbm.b.copy()
    if theta>0:
        while newK>targetK:
            #~ print "K=", newK
            #~ if newK<300:
                #~ iterations=1
            #~ sbm=SBMmh(E,M,K=int(newK),theta=theta,burnin=iterations/2,iterations=iterations,c=c,b=b)
            sbm=sbmModel(E,M,K=int(newK),theta=1,burnin=iterations/2,iterations=iterations,c=c,b=b)
            sbm.infer()
            #~ iterations+=2
            newK=int(newK/sigmaK)
            if newK<targetK:
                newK=targetK
            if int(newK)==targetK:
                iterations=200
            sbm.merge(int(newK))
            c=sbm.c.copy()
            b=sbm.b.copy()
            #~ print np.sum((sbm.drs/np.outer(sbm.nr,sbm.nr))>1)
            #~ iterations*=2
    else :
        newK=targetK
    sbm=sbmModel(E,M,K=newK,theta=theta,burnin=iterations/2,iterations=iterations,c=c,b=b)
    sbm.infer()
    sbm=sbmModel(E,M,K=newK,theta=theta,burnin=iterations/2,iterations=iterations,c=sbm.ccount.argmax(1),b=sbm.bcount.argmax(1))
    #~ print "Log likelihood=",-sbm.calcEntropy(-1)
    return sbm

    
# function to greedily match sbm community labels to metadata labels with
# the greatest overlap.
def greedyLabelMatching(M,C):
    newC=C.copy()
    K=len(np.unique(C))
    
    KK=np.zeros((K,K),dtype=int)
    
    for km in range(K):
        for kc in range(K):
            KK[km,kc]=np.sum((M==km)*(C==kc))
    
    Minds=range(K)
    Cinds=range(K)
    
    #~ print KK
    
    for k in range(K):
        idx=np.argmax(KK)
        m=idx/K
        c=np.mod(idx,K)
        newC[C==c]=m
        #~ print c, "to", m, "(",np.max(KK),")"
        KK[m,:]=-1
        KK[:,c]=-1
    return newC




"""Fits the standard SBM
"""
def fitSBM(E,K,n,greedy_runs=20):
    b=np.ones(n)
    M=np.ones(n)
    minLL=np.inf
    
    for i in range(greedy_runs):
        sbm=greedyAgglom(E,M,targetK=K,initK=n,theta=1,sigmaK=1.1)
        sbm.calcEntropy(-1)
        if sbm.minEntropy<minLL:
            minLL=sbm.minEntropy
            c=sbm.c.copy()
            b=sbm.b.copy()
            #~ with open('%s_sbm_labels_%i.txt' % (network,K),'w') as labelfile:
                #~ for label in c:
                    #~ labelfile.write('%i \n' % label)
        print "***min entropy*** =", minLL, sbm.minEntropy
    return c
    

def loadAndFitSBM(network,meta,greedy_runs=20,K=None):
    E,M,c=loadNetwork(meta,network,c=None)
    if K is None:
        K=len(np.unique(M))
    n=len(M)
    c = fitSBM(E,K,n,greedy_runs=greedy_runs)
    writePartition("%s_SBM%i.txt" % (network,K),c)
    


"""Runs complete algorithm - loads network and runs greedy algorithm multiple times to find
SBM max partition and then runs neoSBM inference for each value of theta
"""
def run_full(meta='c',network='synth2',thetas=np.arange(0,0.011,0.001),c=None,greedy_runs=4,sbmModel=SBMmh,iterations=100,savePartitions=False):
    LLs=[]
    E,M,c=loadNetwork(meta,network,c=c)
    N=len(M)
    print "network loaded"
    targetK=len(np.unique(M))
    b=np.ones(len(M))
    minLL=np.inf
    sbmtext={SBMmh:"SBM", DCSBMmh:"DC"}[sbmModel]
    if c is None:
        #try to load previous SBM partition
        try:
            with open("%s_%s%i.txt" % (network,sbmtext,targetK)) as f:
                c=np.int32(f.readlines())
            print 'c loaded', np.min(c)
        #fit the SBM if no previously saved SBM parition
        except IOError:
            print 'no previously saved partition'
            for i in range(greedy_runs):
                sbm=greedyAgglom(E,M,targetK=targetK,initK=min([600,len(M)]),theta=1,sigmaK=1.1,sbmModel=sbmModel)
                sbm.calcEntropy(-1)
                if sbm.minEntropy<minLL:
                    minLL=sbm.minEntropy
                    c=sbm.c.copy()
                    b=sbm.b.copy()
                    with open('%s_sbm_labels_%i.txt' % (network,targetK),'w') as labelfile:
                        for label in c:
                            labelfile.write('%i \n' % label)
                print "***min entropy*** =", minLL, sbm.minEntropy
        
            with open("%s_%s%i.txt" % (network,sbmtext,targetK),'w') as f:
                for ci in c:
                    f.write('%i \n' % ci)
        
    
    c=greedyLabelMatching(M,c)
    
    
    max_iter=iterations
    
    prs=[np.ones((targetK,targetK))]
    nrs=[np.ones(targetK)]
    modeldict={}
    qs=np.zeros(len(thetas))
    LL=np.zeros(len(thetas))-np.inf
    neoLL=np.zeros(len(thetas))-np.inf
    
    for thi,theta in enumerate(thetas):
        max_iter = iterations
        print "theta=",theta#, np.sum(b,0), c[::125]
        #~ minLL=-np.inf
        bt=b.copy()
        ct=c.copy()
        #~ if theta==thetas[-1]:
        if not (theta>0):
            max_iter=1
        if not theta<1:
            bt[:]=1
            max_iter=0
        sbm=sbmModel(E,M,K=targetK,thetas=thetas, theta_idx=thi,burnin=iterations/2,iterations=max_iter,c=ct,b=bt)
        sbm.infer()
        #~ neoLL.append(sbm.max_neoLL)
        LLth=-sbm.minEntropy
        
        with open('out/%s-%s_partitions.txt' % (network,meta), 'a') as fp:
            for thiw,thetaw in enumerate(thetas):
                fp.write('%e %f %f '  % (thetaw, sbm.max_neoLL[thiw], LLth))
                for ci in sbm.cmax[:,thiw]:
                    fp.write('%i ' % ci)
                fp.write('\n')
        
        #~ print theta,LLt
        #~ if theta==thetas[-1]:
            #~ print "Log likelihood=",LLt
        #~ minLL=LLt
        #~ drst=sbm.drs
        #~ nrt=sbm.nr
        
        new_max_idxs = (neoLL<sbm.max_neoLL).nonzero()[0]
        neoLL[new_max_idxs] = sbm.max_neoLL[new_max_idxs]
        qs[new_max_idxs] = np.sum(sbm.bmax[:,new_max_idxs],0)
        #~ LL[new_max_idxs] = sbm.max_neoLL[new_max_idxs]
        
        #~ LL.append(minLL)
        #~ qs.append(qt)
        #~ print "theta",theta,"SBMLL",LLt,qt, 
        print "neoSBMLL",neoLL
        print "q", qs
        #~ print "SBMLL",neoLL - (qs*np.log(thetas) + (N-qs)*np.log(1-thetas))
        print "SBMLL",sbm.max_LL
        #~ print sbm.cmax
        #~ plt.semilogx(thetas,neoLL- (qs*np.log(thetas/(1-thetas)) + N*np.log(1-thetas)))
    
    LL= neoLL - (qs*np.log(thetas) + (N-qs)*np.log(1-thetas))
    
    
    
    return LL,qs,neoLL

#~ from matplotlib import pyplot as plt
#~ plt.ion()


"""Runs the neoSBM multiple times to find the average LL and qs paths
"""
def run(meta,network,thetamin,greedy_runs=20,sbmModel=SBMmh,iterations=200,runs=1,writeAll=True,savePartitions=True):
    thetas=np.append(0,10**np.arange(thetamin,0-thetamin/50.,-thetamin/50.))
    #~ thetas=np.append(0,10**np.arange(thetamin,np.log10(0.5),-thetamin/50.))
    
    LLs=0
    qqs=0
    sbm={SBMmh:"", DCSBMmh:"DC_"}[sbmModel]
    for i in xrange(runs):
        print i
        LL,qs,neoLL = run_full(meta,network,thetas,greedy_runs=greedy_runs,sbmModel=sbmModel,iterations=iterations,savePartitions=savePartitions)
        LLs+=np.array(LL)
        qqs+=np.array(qs)
        
        with open('out/%s_%s_%sneoLL.txt' % (network,str(meta),sbm),'a') as f:
            f.write('%f %i ' %(thetamin,i))
            for L in neoLL:
                f.write('%f ' % L)
            f.write('\n')
        with open('out/%s_%s_%sLL.txt' % (network,str(meta),sbm),'a') as f:
            f.write('%f %i ' %(thetamin,i))
            for L in LL:
                f.write('%f ' % L)
            f.write('\n')
        with open('out/%s_%s_%sq.txt' % (network,str(meta),sbm),'a') as f:
            f.write('%f %i ' %(thetamin,i))
            for q in qs:
                f.write('%f ' % q)
            f.write('\n')
        



"""Runs search algorithm - loads network and SBM max partition and 
then runs neoSBM inference for by branching values of theta to search for 
distinct LL models.
"""
def run_search(meta='c',network='synth2',log10theta_min=-50,E=None,M=None,c=None,theta_res=0.1):
    LLs=[]
    E,M,c=loadNetwork(meta,network,E,M,c)
    print "network loaded"
    targetK=len(np.unique(M))
    if c is None:
        c=fitSBM(E,targetK,len(M))
    b=np.ones(len(M))
    #~ minLL=np.inf
    #~ if c is None:
        #~ for i in range(greedy_runs):
            #~ sbm=greedyAgglom(E,M,targetK=targetK,initK=600,theta=1,sigmaK=1.1)
            #~ sbm.calcEntropy(-1)
            #~ if sbm.minEntropy<minLL:
                #~ minLL=sbm.minEntropy
                #~ c=sbm.c.copy()
                #~ b=sbm.b.copy()
                #~ with open('%s_sbm_labels_%i.txt' % (network,targetK),'w') as labelfile:
                    #~ for label in c:
                        #~ labelfile.write('%i \n' % label)
            #~ print "***min entropy*** =", minLL, sbm.minEntropy
        
    
    c=greedyLabelMatching(M,c)
    
    model_info={"LLs":[],"thetas":[],"partitions":{}}
    minLL=None
    maxLL=None
    model_info = divide(E,M,targetK,b,c,log10theta_min,0.,model_info,minLL,maxLL,theta_res=theta_res,filestem=network+str(meta))
    return model_info
    

def divide(E,M,K,b,c,log10theta_min,log10theta_max,model_info,minLL=None,maxLL=None,theta_res=0.1,filestem=''):
    
    iterations=200
    
    ### Calculate min and max LL ###
    minLLt=minLL
    maxLLt=maxLL
    if minLL is None:
        bt=b.copy()
        ct=c.copy()
        log10theta=log10theta_min
        sbm=SBMmh(E,M,K,theta=10**log10theta,burnin=iterations/2,iterations=iterations,c=ct,b=bt)
        sbm.infer()
        sbm_min=SBMmh(E,M,K,theta=10**log10theta,burnin=iterations/2,iterations=iterations,c=sbm.ccount.argmax(1),b=sbm.bcount.argmax(1))
        minLLt=-sbm_min.calcEntropy(-1)
        model_info["LLs"].append(minLLt)
        model_info["thetas"].append(log10theta_min)
        if not model_info["partitions"].has_key(minLLt):
            model_info["partitions"][minLLt]=sbm_min.c
        writePartition(filestem+'%.2f' % log10theta_min,sbm_min.c)
        print "ADD MIN"
        print minLLt,log10theta_min,len(model_info["thetas"])
    print log10theta_min,"\tMin Log likelihood=", minLLt
    
    if maxLL is None:
        bt=b.copy()
        ct=c.copy()
        log10theta=log10theta_max
        sbm=SBMmh(E,M,K,theta=10**log10theta,burnin=iterations/2,iterations=iterations,c=ct,b=bt)
        sbm.infer()
        sbm_max=SBMmh(E,M,K,theta=10**log10theta,burnin=iterations/2,iterations=iterations,c=sbm.ccount.argmax(1),b=sbm.bcount.argmax(1))
        maxLLt=-sbm_max.calcEntropy(-1)
        if not model_info["partitions"].has_key(maxLLt):
            model_info["partitions"][maxLLt]=sbm_max.c
    print log10theta_max,"\tMax Log likelihood=", maxLLt
    
    if minLLt==maxLLt:
        #~ print "ADD TERM"
        #~ model_info["LLs"].append(maxLLt)
        #~ model_info["thetas"].append(log10theta_max)
        #~ print maxLLt,log10theta_max,len(model_info["thetas"])
        return model_info
    
    ### Start recursion ###
    if (log10theta_max-log10theta_min) > theta_res:
        log10theta_mid=log10theta_min + (log10theta_max-log10theta_min)/2.
        
        model_info_a = divide(E,M,K,b,c,log10theta_min,log10theta_mid,model_info,minLLt,None,theta_res,filestem)
        
        midLL=model_info_a["LLs"][-1]
        print "modelB",midLL
        
        model_info_b = divide(E,M,K,b,c,log10theta_mid,log10theta_max,model_info,midLL,maxLLt,theta_res,filestem)
        
    else:
        print "ADD RES"
        model_info["LLs"].append(minLLt)
        model_info["thetas"].append(log10theta_min)
        model_info["LLs"].append(maxLLt)
        model_info["thetas"].append(log10theta_max)
        print maxLLt,log10theta_max,len(model_info["thetas"])
        writePartition(filestem+'%.3f' % log10theta_max,model_info["partitions"][maxLLt])
        return model_info
        
    return model_info

def writePartition(file,c):
    with open(file,'w') as f:
        for ci in c:
            f.write('%i \n' % ci)
    




def loadNetwork(meta='c',network='synth',E=None,M=None,c=None):
    if E is None:
        if network=='karate':
            targetK=2
            with open('data/karate_edges_.txt') as f:
                E=np.int32([row.strip().split() for row in f.readlines()])-1
            with open('data/karate_labels_%s.txt' % meta) as f:
            #~ with open('karate_labels_sbm.txt') as f:
                M=np.int32([row.split()[1] for row in f.readlines()])-1
        elif network.startswith('laz'):
            E,M=getLazega(meta,network.strip('laz'))
            targetK=3
        elif network=='seafoodweb':
            with open('data/seafoodweb-edges_undir.txt') as f:
                E=np.int32([row.strip().split()[:2] for row in f.readlines()])
            if meta=='feed':
                with open('data/seafoodweb-feeding.txt') as f:
                    M=np.int32([row.split()[0] for row in f.readlines()])
            else:
                with open('data/seafoodweb-habitat.txt') as f:
                    M=np.int32([row.split()[0] for row in f.readlines()])
        elif network=='airport':
            with open('data/airport-edges.txt') as f:
                E=np.int32([row.strip().split()[:2] for row in f.readlines()])
            if meta=='tmzn':
                with open('data/airport-timezone.txt') as f:
                    M=np.int32([row.split()[0] for row in f.readlines()])
                with open('data/airport_sbm_labels_39_51932.txt') as f:
                    c=np.int32([row.split()[0] for row in f.readlines()])
            else:
                with open('data/airport-country.txt') as f:
                    M=np.int32([row.split()[0] for row in f.readlines()])
                with open('data/airport_sbm_labels_240_33966.txt') as f:
                    c=np.int32([row.split()[0] for row in f.readlines()])
        elif network.startswith('malaria'):
            with open('data/%s.txt' % network) as f:
                E=np.int32([row.strip().split()[:2] for row in f.readlines()])-1
            with open('data/malaria-%s_labels.txt' % meta) as f:
                M=np.int32([row.split()[0] for row in f.readlines()])
            if np.min(M)==1:
                M-=1
        else:
            targetK=4
            if meta=='a':
                with open('%s_local_coreper_labels2.txt' % network) as f:
                    c=np.int32(f.readlines())
                with open('%s_local_assort_labels.txt' % network) as f:
                    M=np.int32(f.readlines())
            else:
                with open('%s_local_assort_labels.txt' % network) as f:
                    c=np.int32(f.readlines())
                with open('%s_local_coreper_labels2.txt' % network) as f:
                    M=np.int32(f.readlines())
            
            with open('%s.txt' % network) as f:
                E=np.int32([row.strip().split() for row in f.readlines()])
    return E,M,c


def getLazega(idx,network='advice'):
    """
    0. seniority
    1. status (1=partner; 2=associate)
    2. gender (1=man; 2=woman)
    3. office (1=Boston; 2=Hartford; 3=Providence)
    4. years with the firm
    5. age
    6. practice (1=litigation; 2=corporate)
    7. law school (1: harvard, yale; 2: ucon; 3: other)"""
    
    if network=='work':
        with open("data/ELwork.dat") as f:
            A=np.int32([row.split() for row in f.readlines() if row.strip()!=''])
    elif network=='advice':
        with open("data/ELadv.dat") as f:
            A=np.int32([row.split() for row in f.readlines() if row.strip()!=''])
    else :
        with open("data/ELfriend.dat") as f:
            A=np.int32([row.split() for row in f.readlines() if row.strip()!=''])
    E=[]
    #~ print "A",A
    for i,row in enumerate(A):
        for j,val in enumerate(row):
            if (val>0) and (i<j):
                E.append([i,j])
    E=np.array(E)
    #~ print E
    with open("data/ELattr.dat") as f:
        M=np.int32([row.split()[idx].strip("\n") for row in f.readlines()])-1
    return E,M