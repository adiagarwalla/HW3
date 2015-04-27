import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
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
    trainMatrix = np.zeros((444075, 444075))
    binMatrix = np.zeros((444075, 444075))

    for line in trainTripletsFile:
        arr = line.split()
        row.append(int(float(arr[0])))
        col.append(int(float(arr[1])))
        dat.append(int(float(arr[2])))
        datBin.append(1)
        #Manually construting the train,binary matrix
        trainMatrix[int(float(arr[0]))][int(float(arr[1]))] = int(float(arr[2]))
        binMatrix[int(float(arr[0]))][int(float(arr[1]))] = 1

    for i in range(444075):
        if binMatrix[0][i] == 1:
            print i


    # Test file reading
    testRow = []; testCol = []; testDat = []
    for line in testTripletsFile:
        arr = line.split()
        testRow.append(int(float(arr[0])))
        testCol.append(int(float(arr[1])))
        testDat.append(int(float(arr[2])))

    # bag = []
    # prev = 0
    # count = 0;

    #adjacency representation
    # for line in trainTripletsFile:
    #     arr = line.split()
    #     if int(float(arr[0])) != prev:
    #         bag.append(docBag)
    #         docBag = []
    #         docBag.append(int(float(arr[1])))
    #         prev = int(float(arr[0]))
    #     else:
    #         if count == 0:
    #             docBag = []
    #             docBag.append(int(float(arr[1])))
    #             count = count + 1
    #         else:
    #             docBag.append(int(float(arr[1])))

    # ACount = csc_matrix((dat, (row, col)), shape=(444075, 444075)).todense()
    # ABin = csc_matrix((datBin, (row, col)), shape=(444075, 444075))
    # # #ut, s, vt = sparsesvd(ABin, 11)

    # #What the R code is doing. 
    # # u = np.transpose(ut)
    # # v = np.transpose(vt)
    # # for i in range(numLinesinTest):
    # #     row = u[testRow[i]]
    # #     col = v[testCol[i]]
    # #     x = np.multiply(row, s)
    # #     p = np.multiply(x, col)

    ABinLDA = csr_matrix((datBin, (row, col)), shape=(444075, 444075))
    ACountRow = csr_matrix((dat, (row, col)), shape=(444075, 444075))
    Test = csr_matrix((testDat, (testRow, testCol)), shape=(444075, 444075))

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
    
        # Check if the sum across all vocab for a topic is ~1
        # for n in range(5):
        #     sum_pr = sum(topic_word[n,:])
        #     print("topic: {} sum: {}".format(n, sum_pr))
    
        n = 15
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
            print topic_words
    
        doc_topic = model.doc_topic_
        print("type(doc_topic): {}".format(type(doc_topic)))
        print("shape: {}".format(doc_topic.shape))
    
        for n in range(10):
            print doc_topic[n]
    
        model.fit(ACountRow)
        topic_word = model.topic_word_
        doc_topic = model.doc_topic_
    
        results = []
    
        for i, value in enumerate(testRow):
            sum = 0
            for k in range(20):
                sum += doc_topic[i][k] * topic_word[k][testCol[i]]
            results.append(sum)
    
        print results
        if len(results) != 10000:
            print "lol"
        # print("type(topic_word): {}".format(type(topic_word)))
        # print("shape: {}".format(topic_word.shape))
    
        # for i, topic_dist in enumerate(topic_word):
        #     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        #     print topic_words
    
        # print("type(doc_topic): {}".format(type(doc_topic)))
        # print("shape: {}".format(doc_topic.shape))
    
        # for n in range(10):
        #   print doc_topic[n]
    
    #Performing KMeans-------------------
    #TODO: Try finding best values for k
    from sklearn.cluster import KMeans
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

    #NMF - used instead of factor analysis because we run out of memory
    from sklearn.decomposition import ProjectedGradientNMF
    if args[0] == "-nmf":
        nmf = ProjectedGradientNMF(n_components=1000, init='random', random_state=0, sparseness='data')
        nmf.fit(ACountRow)
        print "nmf components: "
        print nmf.components_
        print "nmf shape: " + str(nmf.components_.shape)
        print "nmf reconstruction_err: " + str(nmf.reconstruction_err_)

    


if __name__ == "__main__":
    main(sys.argv[1:])