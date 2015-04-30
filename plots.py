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
def ROC(yTest, cmetric):
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

def main(args):
    fileName = args[0]
    probFile = open(fileName, 'rU')
    testTripletsFile = open('testTriplets.txt', 'rU')

    probs = []
    for line in probFile:
        probs.append(line.split()[0])

    # Test file reading
    testRow = []; testCol = []; testDat = []
    for line in testTripletsFile:
        arr = line.split()
        testRow.append(int(float(arr[0])))
        testCol.append(int(float(arr[1])))
        testDat.append(int(float(arr[2])))

    fpr, tpr = ROC(testDat, probs)
    print "FPR: " + str(fpr)
    print "TPR: " + str(tpr)

    precision, recall, thresholds = pr(testDat, probs)
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "thresholds" + str(thresholds)


if __name__ == "__main__":
    main(sys.argv[1:])