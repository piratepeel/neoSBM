import numpy as np

networkFiles = {
                                'malaria_1'   :  'data/malaria_1.txt' ,
                                'malaria_6'   :  'data/malaria_6.txt' ,
                                'synth2'   :  'synth2.txt' 
                            }

metadataFiles = {
                                'ups'   :   'data/malaria-ups_labels.txt',
                                'net1SBM3'   :   'data/malaria-net1SBM3_labels.txt',
                                'c'   :   'synth2_local_coreper_labels2.txt'
                                }



"""
Loads metadata and network
First attempts dictionary lookup for filenames,
then tries to open files directly
"""
def load(network,meta):
    try:
        with open(networkFiles[network]) as f:
            E=np.int32([row.strip().split()[:2] for row in f.readlines()])
    except KeyError:
        with open(network) as f:
            E=np.int32([row.strip().split()[:2] for row in f.readlines()])
    
    if np.min(E)>0:
        E-=np.min(E)
    
    try:
        with open(metadataFiles[meta]) as f:
            M=np.int32([row.split()[0] for row in f.readlines()])
    except KeyError:
        with open(meta) as f:
            M=np.int32([row.split()[0] for row in f.readlines()])
    
    if np.min(M)>0:
        M-=np.min(M)
    
    return E,M
    
    