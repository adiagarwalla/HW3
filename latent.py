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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

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
    # print ABinLDA.shape
    ACountRow = csr_matrix((dat, (row, col)), shape=(444075, 444075))
    Test = csr_matrix((testDat, (testRow, testCol)), shape=(444075, 444075))

    #Performing LDA over range of topics----------------------
    if args[0] == "-t":
        for topics in range(10, 51, 5):
            print topics
            model = lda.LDA(n_topics=topics, n_iter=100, random_state=1)
            model.fit(ACountRow)
            print model.loglikelihood()


    #Performing LDA--------------------
    if args[0] == "-l":
        # x = lda.utils.matrix_to_lists(ACountRow)
        # print x[0].shape
        # print x[1].shape
    
        # model.fit(ACountRow)
    
        vocab = []
        for i in range(444075):
            vocab.append(i)
    
        # topic_word = model.topic_word_
        # print("type(topic_word): {}".format(type(topic_word)))
        # print("shape: {}".format(topic_word.shape))
    
        # Check if the sum across all vocab for a topic is ~1
        # for n in range(5):
        #     sum_pr = sum(topic_word[n,:])
        #     print("topic: {} sum: {}".format(n, sum_pr))
    
        # n = 15
        # for i, topic_dist in enumerate(topic_word):
        #     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        #     # print topic_words
    
        # doc_topic = model.doc_topic_
        # # print("type(doc_topic): {}".format(type(doc_topic)))
        # print("shape: {}".format(doc_topic.shape))
    
        # for n in range(10):
        #     # print doc_topic[n]
    
        model = lda.LDA(n_topics=15, n_iter=100)
        model.fit(ACountRow)
        topic_word = model.topic_word_
        doc_topic = model.doc_topic_
    
        results = []
    
        for i, value in enumerate(testRow):
            sumC = 0
            for k in range(15):
                sumC += doc_topic[value][k] * topic_word[k][testCol[i]]
            results.append(sumC)
    
        print results
        #print model.loglikelihood()
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

    #Performing cosine similarity-----------
    if args[0] == "-c":
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(ACountRow)
        print tfidf_matrix.shape
        results = []
        # x = cosine_similarity(tfidf_matrix[30:31], tfidf_matrix)
        # max_thousand_index = np.argsort(x[0])[-26:][::-1]
        # max_thousand_index_new = max_thousand_index[1:]
        # max_thousand = heapq.nlargest(26, x[0])
        # new_max_thousand = max_thousand[1:]
        # print max_thousand_index
        # print max_thousand


        for i, value in enumerate(testRow):
            # print value
            x = cosine_similarity(tfidf_matrix[value:value+1], tfidf_matrix)
            max_thousand_index = np.argsort(x[0])[-26:][::-1]
            max_thousand_index_new = max_thousand_index[1:]
            max_thousand = heapq.nlargest(26, x[0])
            new_max_thousand = max_thousand[1:]
            max_thousand_norm = [float(i) / sum(new_max_thousand) for i in new_max_thousand]
            sumPredict = 0
            for ind, cos_k in enumerate(max_thousand_index_new):
                if cos_k != value:
                    # print cos_k, testCol[int(i)]
                    if ABinLDA[cos_k, testCol[int(i)]] != 0:
                        sumPredict += max_thousand_norm[ind]

            results.append(sumPredict)

        #             results.append(ABinLDA[max_largest_index[1], testCol[int(i)]])
        bigCount = 0
        for l in range(10000):
            print results[l]
            if (results[l] > 0.5):
                bigCount += 1
            
        print len(results)
        print bigCount
        




if __name__ == "__main__":
    main(sys.argv[1:])