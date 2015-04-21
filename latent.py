import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import pylab as pl
import sys
import math

def main(args):
    trainTripletsFile = open('txTripletsCounts.txt', 'rU')
    trainTriplets = []
    bin = np.zeros((444075, 444075)) # Binary matrix 
    count = np.zeros((444075, 444075)) # Count matrix
 
    for line in trainTripletsFile:
        trainTriplets.append(line.split())

    # the last column in trainTripletsFile is always > 0. 
    for row in trainTriplets:
        bin[int(float(row[0]))][int(float(row[1]))] = 1
        count[int(float(row[0]))][int(float(row[1]))] = row[2]

    binSparse = csr_matrix(bin)


if __name__ == "__main__":
    main(sys.argv[1:])