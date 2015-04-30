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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import nimfa
import pylab as pl
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist


#Function to plot precision_recall curve
def pr(y_true, y_prob):
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.show()

    return (precision, recall, thresholds)

#Function for plotting ROC curves
def ROC(cmetric, yTest):
    fpr, tpr, thresholds = roc_curve(yTest, cmetric[:, 1])
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc

    #Plotting the ROC
    '''pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()'''

    return (fpr, tpr)

#Function for violinp lot
def violin_plot(ax,data,groups,bp=False):
    '''Create violin plot along an axis'''
    dist = max(groups) - min(groups)
    w = min(0.15*dist,0.5)
    for d,p in zip(data,groups):
        k = stats.gaussian_kde(d) #calculates the kernel density
        m = k.dataset.min() #lower bound of violin
        M = k.dataset.max() #upper bound of violin
        x = np.arange(m,M,(M-m)/100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v/v.max()*w #scaling the violin to the available space
        ax.fill_betweenx(x,p,v+p,facecolor='y',alpha=0.3)
        ax.fill_betweenx(x,p,-v+p,facecolor='y',alpha=0.3)
    if bp:
        ax.boxplot(data,notch=1,positions=pos,vert=1)

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
    
    #Choosing K in KMeans
    if args[0] == "-kC":
        K = [10, 50, 100, 200, 250]
        print "Fitting the data..."
        k_means_var = [KMeans(n_clusters = k).fit(ABinLDA) for k in K]
        print "Extracting the centroids..."
        inertias = [(X.inertia_ / 444075) for X in k_means_var] #averaged
        print inertias

        plt.plot(K, inertias)
        plt.show()


    #Performing KMeans-------------------
    if args[0] == "-k":
        n_clusters = 10

        k_means = KMeans(n_clusters)
        k_means.fit(ABinLDA)
        labels = k_means.labels_
        centers = k_means.cluster_centers_

        probInteraction = []

        print "------"
        print len(row)
        print labels.shape
        print labels
        print centers.shape

        for j in range(len(testRow)):
            label = labels[testRow[j]]
            meanInteraction = centers[label]
            if meanInteraction[testCol[j]] > 0.2:
                print meanInteraction[testCol[j]]
            probInteraction.append(meanInteraction[testCol[j]])

        zeroProb = []
        oneProb = []
        #Let's partition the probabilites into 0 and 1 and make the violin plot
        for i in range(len(testDat)):
            if testDat[i] == 0:
                zeroProb.append(probInteraction[i])
            else:
                oneProb.append(probInteraction[i])

        #Function to draw the violin plots
        # groups = range(2)
        # a = np.array(zeroProb)
        # b = np.array(oneProb)
        # data = []
        
        # data.append(a)
        # data.append(b)
        # fig = pl.figure()
        # ax = fig.add_subplot(111)
        # violin_plot(ax,data,groups,bp=0)
        # pl.show()

        pr(testDat, probInteraction)

    #NMF - used instead of factor analysis because we run out of memory
    #from sklearn.decomposition import ProjectedGradientNMF
    if args[0] == "-nmf":
        # nmf = ProjectedGradientNMF(n_components=1000, init='random', random_state=0, sparseness='data')
        # nmf.fit(ACountRow)
        # print "nmf components: "
        # print nmf.components_
        # print "nmf shape: " + str(nmf.components_.shape)
        # print "nmf reconstruction_err: " + str(nmf.reconstruction_err_)
        
        nmf = nimfa.Nmf(ACountRow)#, max_iter=10, rank=2)#, update='euclidean', objective='fro')
        nmf_fit = nmf()
        
        #W = nmf_fit.basis()
        #print('Basis matrix:\n%s' % W.todense())
        
        # H = nmf_fit.coef()
        # print('Mixture matrix:\n%s' % H.todense())
        
        #print('Euclidean distance: %5.3f' % nmf_fit.distance(metric='euclidean'))
        
        # sm = nmf_fit.summary()
        # print('Sparseness Basis: %5.3f  Mixture: %5.3f' % (sm['sparseness'][0], sm['sparseness'][1]))
        # print('Iterations: %d' % sm['n_iter'])
        #print('Target estimate:\n%s' % np.dot(W.todense(), H.todense()))

    #GMM - don't know if this is the best method but might as well give it a try
    #Assuming Gaussian is probably not the best idea but what else are we going to do? YOLO
    if args[0] == "-bmf":
        bmf = nimfa.Bmf(ABinLDA)
        bmf_fit = bmf()

    #Performing cosine similarity-----------
    if args[0] == "-c":
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(ACountRow)
        print tfidf_matrix.shape
        results = []
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