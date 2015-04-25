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

    #LDA
    ABinLDA = csr_matrix((datBin, (row, col)), shape=(444075, 444075))
    ACountLDA = csr_matrix((dat, (row, col)), shape=(444075, 444075))

    model = lda.LDA(n_topics=20, n_iter=1, random_state=1)
    model.fit(ABinLDA)

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

    model.fit(ACountLDA)
    topic_word = model.topic_word_
    print("type(topic_word): {}".format(type(topic_word)))
    print("shape: {}".format(topic_word.shape))

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        print topic_words

    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))

    for n in range(10):
        print doc_topic[n]

    # Test file reading
    # testRow = []; testCol = []; testDat = []
    # for line in testTripletsFile:
    #     arr = line.split()
    #     testRow.append(int(float(arr[0])))
    #     testCol.append(int(float(arr[1])))
    #     testDat.append(int(float(arr[2])))


if __name__ == "__main__":
    main(sys.argv[1:])