from __future__ import print_function
import sys
import random
import os
import numpy as np
import itertools
import scipy.special as sp
from scipy.sparse import csr_matrix
import networkx as nx
import pandas as pd

from sklearn import  preprocessing
from sklearn.model_selection import train_test_split

# ---- Data generation, saving, loading and modification routines

def load_data(input_file):
    mydata = np.loadtxt(input_file,dtype=float)
    return mydata
    
def save_sparse_csr(filename,array):
    np.savez(filename, data = array.data, indices=array.indices, indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename+".npz")
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_matrix(m, output):
    f = open(output, "w")
    for i in range(len(m)):
        print(" ".join([str(x) for x in m[i]]), file=f)
    f.close()

def save_vector(m, output):
    f = open(output, "w")
    for i in range(len(m)):
        print("%5.3f" %(m[i])+" ", file=f)
    f.close()

# generates a random matrix representing samples from a two-component GMM with identity covariance
def generate_random_matrix_normal(mu1, mu2, n_rows, n_cols):
    ctrmu2 = np.random.binomial(n_rows,0.5)
    ctrmu1 = n_rows - ctrmu2 
    mfac = 10/np.sqrt(n_cols)
    return np.concatenate((np.add(mfac*np.random.standard_normal((ctrmu1, n_cols)), mu1), np.add(mfac*np.random.standard_normal((ctrmu2, n_cols)), mu2)))

# generates a vector of random labels, each entry only has value -1 or 1
def generate_random_binvec(n):
    return np.array([np.random.randint(2)*2-1 for x in range(n)])

def interactionTermsAmazon(data, degree, hash=hash):
    new_data = []
    m,n = data.shape
    for indicies in itertools.combinations(range(n), degree):
        if not(5 in indicies and 7 in indicies) and not(2 in indicies and 3 in indicies):
            new_data.append([hash(tuple(v)) for v in data[:, indicies]])
    return np.array(new_data).T

# ---- Other routines 

def unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

def getB(n_workers,n_stragglers):
    Htemp=np.random.normal(0,1,[n_stragglers,n_workers-1])
    H=np.vstack([Htemp.T,-np.sum(Htemp,axis=1)]).T

    Ssets=np.zeros([n_workers,n_stragglers+1])

    for i in range(n_workers):
        Ssets[i,:]=np.arange(i,i+n_stragglers+1)
    Ssets=Ssets.astype(int)
    Ssets=Ssets%n_workers
    B=np.zeros([n_workers,n_workers])
    for i in range(n_workers):
        B[i,Ssets[i,0]]=1
        vtemp=-np.linalg.solve(H[:,np.array(Ssets[i,1:])],H[:,Ssets[i,0]])
        ctr=0
        for j in Ssets[i,1:]:
            B[i,j]=vtemp[ctr]
            ctr+=1

    return B

def getBapprox(n, d, expander_ver):
    B=np.zeros([n,n])

    # this function defines various expander constructions based on expander_ver

    if(expander_ver==0):
        # expander_ver = 0 represents a random d-regular graph
        # should potentially add Identity to this B matrix ?
        G = nx.random_regular_graph(d,n)
        edge_list = list(G.edges)
        for u,v in edge_list:
            B[u,v] = 1
            B[v,u] = 1

    elif(expander_ver==1):
        #expander_ver = 1 represents a random d-regular (both sides) bipartite graph
        while True:
            budget = d*np.ones(n)
            B[:]=0
            for i in range(n):
                valset = [i for i in range(n) if budget[i]>0]
                perm = np.random.permutation(valset)[:d]
                budget[perm]=budget[perm]-1
                B[i,perm] = 1
            if sum(budget)==0:
                break

    elif(expander_ver==2):
        #expander_ver = 2 represents a Margulis graph (number of vertices n must be square)
        assert((np.sqrt(n)).is_integer())
        G = nx.margulis_gabber_galil_graph(int(np.sqrt(n)))
        G = nx.convert_node_labels_to_integers(G)
        edge_list = G.edges()
        for e in edge_list:
            B[e[0],e[1]]+= 1
            B[e[1],e[0]]+= 1

    return (1.0/d)*B

def getArowlinear(completed_workers):
    # this function defines a linear time reconstruction based on machines that have returned --- (not used in the code though)
    n = len(completed_workers)
    s = n - np.sum(completed_workers)

    w = n/(n-s + 0.0)

    Arow = np.zeros((1,n))
    for i in range(n):
        if completed_workers[i]:
            Arow[0,i] = w

    return Arow

def getA(B,n_workers,n_stragglers):
    #S=np.array(list(itertools.permutations(np.hstack([np.zeros(n_stragglers),np.ones(n_workers-n_stragglers)]),n_workers)))
    #print(S)
    #S=unique_rows(S)
    
    S = np.ones((int(sp.binom(n_workers,n_stragglers)),n_workers))
    combs = itertools.combinations(range(n_workers), n_stragglers)
    i=0
    for pos in combs:
        S[i,pos] = 0
        i += 1

    (m,n)=S.shape
    A=np.zeros([m,n])
    for i in range(m):
        sp_pos=S[i,:]==1
        A[i,sp_pos]=np.linalg.lstsq(B[sp_pos,:].T,np.ones(n_workers))[0]

    return A

def compare(a,b):
    for id in range(len(a)):
        if a[id] and (not b[id]):
            return 1
        if (not a[id]) and b[id]:
            return -1
    return 0

def binary_search_row_wise(Aindex,completed,st,nd):
    if st>=nd-1:
        return st
    idx=(st+nd)/2
    cp=compare(Aindex[idx,:],completed)
    if (cp==0):
        return idx
    elif (cp==1):
        return binary_search_row_wise(Aindex,completed,st,idx)
    else:
        return binary_search_row_wise(Aindex,completed,idx+1,nd)

def calculate_indexA(boolvec):
    l = len(boolvec)
    ctr = 0
    ind = 0
    for j in range(l-1,-1, -1):
        if boolvec[j]:
            ctr = ctr+1
            ind = ind + sp.binom(l-1-j, ctr)

    return int(ind)

def calculate_loss(y,predy,n_samples):
    return np.sum(np.log(1+np.exp(-np.multiply(y,predy))))/n_samples

def load_amazon_data(input_dir, n_procs):
    real_dataset = "amazon-dataset"

    # print("Preparing data for " + real_dataset)
    trainData = pd.read_csv(input_dir + 'train.csv')
    trainX = trainData.ix[:, 'RESOURCE':].values
    trainY = trainData['ACTION'].values

    relabeler = preprocessing.LabelEncoder()
    for col in range(len(trainX[0, :])):
        relabeler.fit(trainX[:, col])
        trainX[:, col] = relabeler.transform(trainX[:, col])

    trainY = 2 * trainY - 1

    d_all_s = interactionTermsAmazon(trainX, degree=2)  # second order
    # d_all_t = interactionTermsAmazon(trainX, degree=3)  # third order
    # trainX = np.hstack((trainX, d_all_s, d_all_t))
    trainX = np.hstack((trainX, d_all_s))

    for col in range(len(trainX[0, :])):
        relabeler.fit(trainX[:, col])
        trainX[:, col] = relabeler.transform(trainX[:, col])

    trainX = np.vstack([trainX.T, np.ones(trainX.shape[0])]).T

    X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.2, random_state=0)

    encoder = preprocessing.OneHotEncoder(sparse=True)
    encoder.fit(np.vstack((X_train, X_valid)))
    X_train = encoder.transform(X_train)  # Returns a sparse matrix (see numpy.sparse)
    X_valid = encoder.transform(X_valid)

    n_rows, n_cols = X_train.shape
    # print("No. of training samples = %d, Dimension = %d" % (n_rows, n_cols))
    # print("No. of testing samples = %d, Dimension = %d" % (X_valid.shape[0], X_valid.shape[1]))

    # Create output directory
    partitions = n_procs - 1

    n_rows_per_worker = n_rows // partitions

    return X_train, y_train, X_valid, y_valid