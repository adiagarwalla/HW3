import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import pylab as pl
import sys
import math

def main(args):
    trainTripletsFile = open('txTripletsCounts.txt', 'rU')
    trainTriplets = []
    row = []; col = []; dat = []; datBin = []

    for line in trainTripletsFile:
        arr = line.split()
        row.append(int(float(arr[0])))
        col.append(int(float(arr[1])))
        dat.append(int(float(arr[2])))
        datBin.append(1)

    ACount = csr_matrix((dat, (row, col)), shape=(444075, 444075))
    ABin = csr_matrix((datBin, (row, col)), shape=(444075, 444075))
    print ABin

if __name__ == "__main__":
    main(sys.argv[1:])