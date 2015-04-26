import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sparsesvd import sparsesvd
import pylab as pl
import matplotlib.pyplot as plt
import sys
import math
import lda

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
    #ABin = csc_matrix((datBin, (row, col)), shape=(444075, 444075))
    #ut, s, vt = sparsesvd(ABin, 11)

    #What the R code is doing. 
    # u = np.transpose(ut)
    # v = np.transpose(vt)
    # for i in range(numLinesinTest):
    #     row = u[testRow[i]]
    #     col = v[testCol[i]]
    #     x = np.multiply(row, s)
    #     p = np.multiply(x, col)

    ABinLDA = csr_matrix((datBin, (row, col)), shape=(444075, 444075))
    ACountRow = csr_matrix((dat, (row, col)), shape=(444075, 444075))

    #Performing LDA--------------------
    if args[0] == "-l":
        # x = lda.utils.matrix_to_lists(ACountRow)
        # print x[0].shape
        # print x[1].shape
    
        model = lda.LDA(n_topics=20, n_iter=1, random_state=1)
        model.fit(ACountRow)
    
        vocab = []
        for i in range(444075):
            vocab.append(i)
    
        topic_word = model.topic_word_
        print("type(topic_word): {}".format(type(topic_word)))
        print("shape: {}".format(topic_word.shape))
    
        for n in range(5):
            sum_pr = sum(topic_word[n,:])
            print("topic: {} sum: {}".format(n, sum_pr))
    
        n = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
            print topic_words
    
    #Performing KMeans-------------------
    if args[0] == "-k":
        n_clusters = 10
        k_means = KMeans(n_clusters)
        k_means.fit(ACountRow)
        labels = k_means.labels_
        centers = k_means.cluster_centers_

        #Printing out the labels for each of the giver addresses
        for i in range(10):
            print labels[i]

        #Printing out the coordinates for the cluster centers
        print centers


if __name__ == "__main__":
    main(sys.argv[1:])