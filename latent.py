import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sparsesvd import sparsesvd
import pylab as pl
import matplotlib.pyplot as plt
import sys
import math

def main(args):
    trainTripletsFile = open('txTripletsCounts.txt', 'rU')
    testTripletsFile = open('testTriplets.txt', 'rU')
    row = []; col = []; dat = []; datBin = []    

    for line in trainTripletsFile:
        arr = line.split()
        row.append(int(float(arr[0])))
        col.append(int(float(arr[1])))
        dat.append(int(float(arr[2])))
        datBin.append(1)

    testRow = []; testCol = []; testDat = []

    numLinesinTest = 0
    for line in testTripletsFile:
        numLinesinTest = numLinesinTest + 1
        arr = line.split()
        testRow.append(int(float(arr[0])))
        testCol.append(int(float(arr[1])))
        testDat.append(int(float(arr[2])))

    #ACount = csc_matrix((dat, (row, col)), shape=(444075, 444075)).todense()
    ABin = csc_matrix((datBin, (row, col)), shape=(444075, 444075))

    ut, s, vt = sparsesvd(ABin, 11)
    # print ut.shape
    print s.shape
    # print vt.shape
    # print np.transpose(ut)
    # print s
    # print np.transpose(vt)
    u = np.transpose(ut)
    v = np.transpose(vt)
    for i in range(numLinesinTest):
        row = u[testRow[i]]
        col = v[testCol[i]][0]
        #print col
        x = np.multiply(row, s)
        p = np.multiply(x, col)
        print np.sum(p)

    
    #print testBinSparse
    print ABin
    svd = TruncatedSVD(n_components=11)
    svd.fit(ABin)
    print(svd.components_)
    print(svd.components_.shape)
    print (svd.explained_variance_ratio_)
    #plt.plot(svd.explained_variance_ratio_)
    

if __name__ == "__main__":
    main(sys.argv[1:])