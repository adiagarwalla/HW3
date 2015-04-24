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

    for line in testTripletsFile:
        arr = line.split()
        testRow.append(int(float(arr[0])))
        testCol.append(int(float(arr[1])))
        testDat.append(int(float(arr[2])))

    testBinSparse = csr_matrix((testDat, (testRow, testCol)), shape=(444075, 444075))


    #ACount = csc_matrix((dat, (row, col)), shape=(444075, 444075)).todense()
    ABin = csc_matrix((datBin, (row, col)), shape=(444075, 444075))

    ut, s, vt = sparsesvd(ABin, 11)
    print ut.shape
    print s.shape
    print vt.shape
    print np.transpose(ut)
    print s
    print np.transpose(vt)
    

    #print testBinSparse


    # print ut
    # print s
    # print vt
    #print ACount
    #print ABin
    #U, s, V = np.linalg.svd(ABin, full_matrices=True)
    # svd = TruncatedSVD(n_components=25)
    # svd.fit(ABin)
    # #print (svd.explained_variance_ratio_)
    # plt.plot(svd.explained_variance_ratio_)
    # plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])