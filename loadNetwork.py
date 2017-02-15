import numpy as np



"""
Loads metadata and network
"""
def load(networkfile,metadatafile):
    with open(networkfile) as f:
        E=np.int32([row.strip().split()[:2] for row in f.readlines()])
    
    if np.min(E)>0:
        E-=np.min(E)
    
    M = loadPartition(metadatafile)
    
    return E,M


def loadPartition(partitionFile):
    
    with open(partitionFile) as f:
        M=np.int32([row.split()[0] for row in f.readlines()])
    
    
    return M