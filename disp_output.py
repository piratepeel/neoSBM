from matplotlib import pyplot as plt
import numpy as np
import loadNetwork
from scipy import sparse
import matplotlib.patches as mpatches
import partitions as sample_parts

plt.ion()

##################################################
"""
Main function to plot neoSBM output
Plots number of free nodes and log likelihood as a function of theta
"""
def plotLq(network,meta,log=True,DC=False):
    plt.figure()
    
    labelx=-.1
    
    ax=plt.subplot(2,1,1)
    qs=plotLL(network,meta,"q",log,DC)
    plt.ylabel("# of free nodes",size=20)
    ax.yaxis.set_label_coords(labelx, 0.5)
    
    ax=plt.subplot(2,1,2)
    plotLL(network,meta,"LL",log,DC,qs=qs)
    plt.ylabel("SBM log likelihood",size=20)
    ax.yaxis.set_label_coords(labelx, 0.5)
    plt.xlabel("$\\theta$",size=20)
    plt.tight_layout()
    if DC:
        plt.ylabel("DCSBM log likelihood",size=20)
        plt.savefig('out/%s_%sDC_lqs.svg' % (network,meta))
    else:
        plt.savefig('out/%s_%s_lqs.svg' % (network,meta))



def plotLL(network,meta,line="LLs",log=False,DC=False,lc='b',qs=None):
    nl="neoLL"
    if DC:
        line="DC_"+line
        nl="DC_" + nl
    with open("out/%s_%s_%s.txt" %(network,str(meta),line)) as f:
        results=np.float64([row.strip().split() for row in f.readlines()])
    
    with open("out/%s_%s_%s.txt" %(network,str(meta),nl)) as f:
        LLs=np.float64([row.strip().split() for row in f.readlines()])
    
    
    thetamin=results[0,0]
    thetas=np.append(0,10**np.arange(thetamin,0-thetamin/50.,-thetamin/50.))
    idx=np.argmax(LLs,0)[2:]
    
    y=results[idx,np.arange(len(thetas))+2]
    #~ print zip(thetas,y)
    if log:
        plt.semilogx(thetas,y,lc)
        #~ try:
            #~ print y[0],qs[-1]
            #~ N=qs[-1]
            #~ ftheta=y[0]-qs*np.log(thetas/(1-thetas)) - N*np.log(1-thetas)
            #~ plt.semilogx(thetas, ftheta,'r:')
            
            #~ for q in np.arange(0.,N,50.):
                #~ print q
                #~ ftheta=y[0]-q*np.log(thetas/(1-thetas)) - N*np.log(1-thetas)
                #~ plt.semilogx(thetas, ftheta,'k:')
            
        #~ except TypeError:
            #~ pass
    else:
        plt.plot(thetas,y,lc)
    plt.xlim(0,1)
    #~ print results[0,1]
    return y
    
##################################################

def plotBlocks(network,meta,thetas):
    
    E,M = loadNetwork.load(network,meta)
    n=len(M)
    m=len(E)
    K=len(np.unique(M))
    A=sparse.coo_matrix((np.ones(m),(E[:,0],E[:,1])),shape=(n,n)).tocsc()
    A=A+A.T
    
    #get partition data
    with open('out/%s-%s_partitions.txt' % (network,meta)) as f:
        partition_data = np.float64([row.split() for row in f.readlines()])
    
    all_thetas= np.unique(partition_data[:,0])
    
    prs_set=[]
    nr_set=[]
    
    Y=sparse.coo_matrix((np.ones(n),(np.arange(n),M)),shape=(n,K)).tocsc()
    crs=Y.T.dot(A.dot(Y)).toarray()
    prs_set.append(crs/n)
    nr_set.append(np.array(Y.sum(0)).flatten())
    
    for theta in thetas:
        #find the partition with closest theta and highest neoLL
        closest_theta=all_thetas[np.argmin(np.abs(all_thetas-theta))]
        print theta, closest_theta
        theta_partitions = (partition_data[:,0]==closest_theta).nonzero()[0]
        partition_index = np.argmax(partition_data[theta_partitions,1])
        Cti=partition_data[theta_partitions[partition_index],3:]
        
        Y=sparse.coo_matrix((np.ones(n),(np.arange(n),Cti)),shape=(n,K)).tocsc()
        crs=Y.T.dot(A.dot(Y)).toarray()
        prs_set.append(crs/n)
        nr_set.append(np.array(Y.sum(0)).flatten())
        
    plt.figure()
    ax=plt.subplot(111)
    
    p_max=np.max([prs.max() for prs in prs_set])
    for pi,(prs,nr) in enumerate(zip(prs_set,nr_set)):
        #~ grid = np.mgrid[0.:1.:N*1j, 0.:1.:N*1j]
        #~ p=prs
        nr/=n
        y=np.append(0,np.cumsum(nr))
        x=1-np.cumsum(nr)
        #~ xy-=xy[0]
        
        rect=[pi/float(len(prs_set)),0,0.1,0.4]
        ax_=add_subplot_axes(ax,rect,axisbg='w')
        
        patches=[]
        colors=[]
        
        for r in range(K):
            for s in range(K):
                rect=mpatches.Rectangle([x[r],y[s]], nr[r], nr[s], ec=np.zeros(3),fill=True,color=np.ones(3)-prs[r,s]/p_max)
                #~ rect=mpatches.Rectangle([x[r],y[s]], n[r], n[s], ec=np.zeros(3),fill=True,color=np.ones(3)+(np.log(p[r,s])-p_min)/p_max)
                ax_.add_patch(rect)
    
    plt.savefig('out/%s_%s_blocks.svg' % (network,meta))
    
    
def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg,aspect='equal')
    #~ x_labelsize = subax.get_xticklabels()[0].get_size()
    #~ y_labelsize = subax.get_yticklabels()[0].get_size()
    #~ x_labelsize *= rect[2]**0.5
    #~ y_labelsize *= rect[3]**0.5
    #~ subax.xaxis.set_tick_params(labelsize=x_labelsize)
    #~ subax.yaxis.set_tick_params(labelsize=y_labelsize)
    plt.axis('off')
    return subax
 

##################################################    
    

def sample_partitions(network,meta):
    
    E,M = loadNetwork.load(network,meta)
    n=len(M)
    m=len(E)
    K=len(np.unique(M))
    
    #get partition data
    with open('out/%s-%s_partitions.txt' % (network,meta)) as f:
        partition_data = np.float64([row.split() for row in f.readlines()])
    
    thetas= np.unique(partition_data[:,0])
    
    partitions=[]
    
    for theta in thetas:
        theta_partitions = (partition_data[:,0]==theta).nonzero()[0]
        partition_index = np.argmax(partition_data[theta_partitions,1])
        partitions.append(partition_data[theta_partitions[partition_index],3:])
        
    
    
    sample_parts.run_partitions(partitions,E,outFile='out/%s_%s_xyz.txt' % (network,meta),nreps=3)




















