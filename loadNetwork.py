import numpy as np
from scipy import sparse

def load(networkfile, metadatafile):
    """
    Loads metadata and network
    """
    with open(networkfile) as f:
        E = np.int32([row.strip().split()[:2] for row in f.readlines()])

    if np.min(E) > 0:
        E -= np.min(E)

    M = loadPartition(metadatafile)

    E = checkNetwork(E)

    return E,M


def loadPartition(partitionFile):

    with open(partitionFile) as f:
        M = np.int32([row.split()[0] for row in f.readlines()])

    return M


def checkNetwork(E):
    # check that there are no self loops
    if np.sum(E[:, 0] == E[:, 1]):
        print "WARNING Network contains self loops, neoSBM may not work as expected"

    # check no multiedges
    data = np.ones(len(E))
    row = E[:,0]
    col = E[:,1]
    n = np.max(E)+1
    A = sparse.coo_matrix((data, (row, col)), shape=(n, n))
    if A.max() > 1:
        print "\n\n\n***WARNING: Network contains multiedges,"
        print "converting to undirected and unweighted"
        newE = np.array(sparse.tril(A+A.T).nonzero()).T
        print len(newE)
        return newE
    else:
        return E
