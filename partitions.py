import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from sklearn import manifold
from sklearn import neighbors
from random import sample
from mpl_toolkits.mplot3d import Axes3D
import os

plt.ion()



def plot2(k=1,xyzFile='xyz_synth_surf.txt',write=False):
    with open(xyzFile) as f:
        xyz=np.float64([row.split() for row in f.readlines()])
    fig=plt.figure()
    #~ ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:,2], marker='o')

def plot(k=1,xyzFile='xyz_synth_surf.txt',write=False):
    with open(xyzFile) as f:
        xyz=np.float64([row.split() for row in f.readlines()])
    
    #~ plt.figure()
    #~ plt.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:,2])
    #~ plt.plot(xyz[:3, 0], xyz[:3, 1], c='k', marker='s',ms=10)
    #~ plt.plot(xyz[:50, 0], xyz[:50, 1], xyz[:50,2], c='k', marker='s',ms=3)
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], c=xyz[:,2], marker='o',linewidths=0)
    ax.plot(xyz[:50, 0], xyz[:50, 1], xyz[:50,2], c='k', marker='s',ms=3)
    #~ ax.scatter(xyz[:50, 0], xyz[:50, 1], xyz[:50,2], c='k', marker='s',linewidths=0,cmap=plt.cm.bone)
    
    xmin=np.min(xyz[:,0])
    xmax=np.max(xyz[:,0])
    step=(xmax-xmin)/100.
    
    x_=np.arange(np.min(xyz[:,0]),np.max(xyz[:,0]),step)
    y_=np.arange(np.min(xyz[:,0]),np.max(xyz[:,0]),step)
    xx,yy=np.meshgrid(x_,y_)
    
    xy=np.append(xx.ravel()[:,np.newaxis],yy.ravel()[:,np.newaxis],1)
    
    knn = neighbors.KNeighborsRegressor(k, weights='distance',p=1)
    z_= knn.fit(xyz[:,:2],xyz[:,2]).predict(xy)
    
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z_.reshape(np.shape(xx)),rstride=1, cstride=1, cmap=plt.cm.spectral,
                       linewidth=0, antialiased=False)
    ax.plot(xyz[:50, 0], xyz[:50, 1], xyz[:50,2], c='k', marker='s',ms=3)
    if write:
        with open('knn_'+ xyzFile,'w') as f:
            for xi,yi,zi in zip(xx.ravel(),yy.ravel(),z_):
                f.write('%f %f %f\n' % (xi,yi,zi))
    
    


#################################################################
#calculate a distance matrix based on variation of information
def calcVI(partitions):
    
    num_partitions,n=np.shape(partitions)
    nodes = np.arange(n)
    c=len(np.unique(partitions[0]))
    vi_mat=np.zeros((num_partitions,num_partitions))
    
    print 'calcvi'
    
    for i in xrange(num_partitions):
        if i%250==0:
            print i
        A1 = sparse.coo_matrix((np.ones(n),(partitions[i,:],nodes)),shape=(c,n),dtype=np.uint).tocsc()
        n1all = np.array(A1.sum(1),dtype=float)
        
        for j in xrange(i):
            
            A2 = sparse.coo_matrix((np.ones(n),(nodes,partitions[j,:])),shape=(n,c),dtype=np.uint).tocsc()
            n2all = np.array(A2.sum(0),dtype=float)
            
            n12all = np.array(A1.dot(A2).todense(),dtype=float)
            
            rows, columns = n12all.nonzero()
            
            vi = np.sum(n12all[rows,columns]*np.log((n12all*n12all/(np.outer(n1all,n2all)))[rows,columns]))
            
            vi = -1/n*vi
            vi_mat[i,j]=vi
            vi_mat[j,i]=vi
    
    print "vi"
    return vi_mat
    
#################################################################
#Perform embedding into a 2D space using MDS
def embedding(vi_mat,LL,n_neighbors=10):
    n_components=2
    Y = manifold.MDS(n_components,dissimilarity='precomputed').fit_transform(vi_mat)
    
    color=np.zeros(1000)
    color[:6]=np.ones(6)
    
    #~ plt.figure()
    #~ plt.plot(Y[:, 0], Y[:, 1], 'k.')
    #~ plt.plot(Y[-n_close:, 0], Y[-n_close:, 1], 'r.')
    #~ for i in xrange(6):
        #~ plt.plot(Y[i, 0], Y[i, 1], 'bo',ms=3+3*i)
    #~ plt.scatter(Y[:, 0], Y[:, 1], c=LL)
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], LL[:,0], c=LL[:,0], marker='o')
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], LL[:,1], c=LL[:,1], marker='o')
    
    return Y

#################################################################
#load known partitions
def getPartitions(dir='../synthetic/partitions/',files=['synth2c-%s' % nn for nn in ['50.00','12.11','22.17','44.34']]):
    
    nf=len(files)
    
    #~ partitions=np.empty((nf,n))
    
    for fi,file in enumerate(files):
        print fi,file
        try:
            with open(dir+file) as f:
                partition=np.int32(f.readlines())
        except:
            with open(dir+file) as f:
                partition=np.int32([row.strip().split() for row in f.readlines()])[:,1]
        if np.min(partition)>0:
            partition-=1
        try:
            partitions[fi,:]=partition
        except:
            n=len(partition)
            partitions=np.empty((nf,n))
            partitions[fi,:]=partition
    
    return partitions


#################################################################
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

#################################################################
#calculate the log likelihood of a partition
def calcLL(partitions,edges):
    
    if np.min(edges)>0:
        edges-=1
    
    thetas=np.log(10**np.arange(-50,0.1,1)+1e-200)
    oneminusthetas=np.log(1-10**np.arange(-50,0.1,1)+1e-200)
    LL=np.empty((np.shape(partitions)[0],2+len(thetas)))
    n=np.shape(partitions)[1]
    m=np.shape(edges)[0]
    ei=np.arange(m)
    K=len(np.unique(partitions[0]))
    
    for pi,partition in enumerate(partitions):
        nk=np.array([np.sum(partition==k) for k in xrange(K)])
        mi=partition[edges[:,0]]
        mj=partition[edges[:,1]]
        A1 = sparse.coo_matrix((np.ones(m),(mi,ei)),shape=(K,m),dtype=np.uint).tocsc()
        A2 = sparse.coo_matrix((np.ones(m),(ei,mj)),shape=(m,K),dtype=np.uint).tocsc()
        
        e_rs = np.array(A1.dot(A2).todense(),dtype=float)
        if not np.all(e_rs==e_rs.T):
            e_rs+=e_rs.T
        nrns = np.outer(nk,nk)
        prs=e_rs/(nrns+1e-200)
        LL[pi,0]= (e_rs*np.log(prs+1e-200)+(nrns-e_rs)*np.log(1-prs+1e-200)).sum()/2.
        
        dr=np.sum(e_rs,0)
        drds=np.outer(dr,dr)
        prs=e_rs/drds
        LL[pi,1]= (e_rs*np.log(prs+1e-200)).sum()/2. 
        #~ print pi,m,e_rs,e_rs.sum(), LL[pi]
        q=n-np.sum(partitions[0,:]==partition)
        LL[pi,2:]= LL[pi,1] + np.float64([(q)*theta + (n-q)*(oneminustheta) for theta,oneminustheta in zip(thetas,oneminusthetas)]) 
    return LL



#################################################################
#main function
def run_partitions(partitions,edges,nreps=15,divisions=20,outFile=None,DC=False):
    n_random=500
    
    c=len(np.unique(partitions))
    n=np.shape(partitions)[1]
    n_partitions=np.shape(partitions)[0]
    partitions=np.append(partitions,np.random.randint(0,c,(n_random,n)),0)
    
    n_close=n_partitions*divisions*nreps
    
    partitions=np.append(partitions,np.zeros((n_close,n)),0)
    
    print np.shape(partitions)
    
    pi=n_partitions+n_random
    reps=range(nreps)
    steps_size=n/divisions
    n_lim=(steps_size)*divisions
    #~ if n%20>0:
        #~ n_lim-=steps_size
    
    print n_partitions,len(range(0,n_lim,steps_size)),n,n_lim,steps_size,n_partitions*len(range(0,n_lim,steps_size))*nreps+n_random+n_partitions
    
    for i in xrange(n_partitions):
        for si in xrange(0,n_lim,steps_size):
            for rep in reps:
                #randomly select a partition j
                j=sample(xrange(n_partitions),1)[0]
                #select labels to fix to partition j
                nj=sample(xrange(n),1)[0]
                inds = sample(xrange(n),nj)
                #combine partitions i and j
                partitions[pi,:] = partitions[i,:]
                partitions[pi,inds] = partitions[j,inds]
                inds = sample(xrange(n),si)
                partitions[pi,inds] = np.random.randint(0,c,(si))
                pi+=1
    
    
    LL=calcLL(partitions,edges)
    print pi
    
    order=sample(xrange(1000),1000)
    
    vi=calcVI(partitions)
    Y=embedding(vi,LL)
    if outFile is not None:
        with open(outFile,'w') as f:
            for yi,zi in zip(Y,LL):
                #~ f.write('%f %f %f %f\n' % (yi[0],yi[1],zi[0],zi[1]))
                f.write('%f %f ' % (yi[0],yi[1]))
                for ll in zi:
                    f.write('%f ' % ll)
                f.write('\n')
    
    return vi,LL,partitions


#################################################################


#################################################################
#main function
def run(path='../synthetic/partitions/',partFiles=['synth2c-%s' % nn for nn in ['50.00','12.11','22.17','44.34']],graphFile='../synthetic/synth2.txt',reps=15,divisions=20,outFile=None,DC=False):
    n_random=500
    
    partitions=getPartitions(dir=path,files=partFiles)
    c=len(np.unique(partitions))
    n=np.shape(partitions)[1]
    n_partitions=np.shape(partitions)[0]
    partitions=np.append(partitions,np.random.randint(0,c,(n_random,n)),0)
    
    n_close=n_partitions*divisions*reps
    
    partitions=np.append(partitions,np.zeros((n_close,n)),0)
    
    print np.shape(partitions)
    
    pi=n_partitions+n_random
    reps=range(reps)
    steps_size=n/divisions
    n_lim=(steps_size)*divisions
    #~ if n%20>0:
        #~ n_lim-=steps_size
    
    print n_partitions,len(range(0,n_lim,steps_size)),n,n_lim,steps_size,n_partitions*len(range(0,n_lim,steps_size))*15+n_random+n_partitions
    
    for i in xrange(n_partitions):
        for si in xrange(0,n_lim,steps_size):
            for rep in reps:
                j=sample(xrange(n_partitions),1)[0]
                nj=sample(xrange(n),1)[0]
                inds = sample(xrange(n),nj)
                partitions[pi,:] = partitions[i,:]
                partitions[pi,inds] = partitions[j,inds]
                inds = sample(xrange(n),si)
                partitions[pi,inds] = np.random.randint(0,c,(si))
                pi+=1
    
    
    LL=calcLL(partitions,file=graphFile)
    print pi
    
    order=sample(xrange(1000),1000)
    
    vi=calcVI(partitions)
    Y=embedding(vi,LL)
    if outFile is not None:
        with open(outFile,'w') as f:
            for yi,zi in zip(Y,LL):
                #~ f.write('%f %f %f %f\n' % (yi[0],yi[1],zi[0],zi[1]))
                f.write('%f %f ' % (yi[0],yi[1]))
                for ll in zi:
                    f.write('%f ' % ll)
                f.write('\n')
    
    return vi,LL,partitions


#################################################################
#run specific examples

def runSynth2():
    return run(path='../synthetic/partitions/',partFiles=['synth2c-%s' % nn for nn in ['12.11','22.17','44.34','50.00']],graphFile='../synthetic/synth2.txt',outFile='xyz_synth2_surf.txt')

def runLazegaF():
    partFiles=os.listdir('../LazegaLawyers/partitions/')
    partFiles.sort()
    return run(path='../LazegaLawyers/partitions/',partFiles=partFiles,graphFile='../LazegaLawyers/lazfriend_edges.txt',reps=1,divisions=5,outFile='xyz_lazfriend_surf.txt')

def runKarate(DC=False):
    partFiles=os.listdir('../karateclub/partitions/')
    partFiles.sort()
    return run(path='../karateclub/partitions/',partFiles=partFiles,graphFile='../karateclub/karate_edges.txt',reps=20,divisions=30,outFile='xyz_karate_surf.txt')





    