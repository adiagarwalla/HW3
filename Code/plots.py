import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import sys
import pylab as pl
from scipy import stats
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc

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

#Function to plot precision_recall curve
def pr(y_true, y_prob):
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # plt.clf()
    # plt.plot(recall, precision, label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.legend(loc="lower left")
    # plt.show()

    return (precision, recall, thresholds)

#Function for plotting ROC curves
def ROC(yTest, cmetric):
    fpr, tpr, thresholds = roc_curve(yTest, cmetric)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc

    #Plotting the ROC
    # pl.clf()
    # pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # pl.plot([0, 1], [0, 1], 'k--')
    # pl.xlim([0.0, 1.0])
    # pl.ylim([0.0, 1.0])
    # pl.xlabel('False Positive Rate')
    # pl.ylabel('True Positive Rate')
    # pl.title('Receiver operating characteristic example')
    # pl.legend(loc="lower right")
    # pl.show()

    return (fpr, tpr)

def main(args):
    fileName = args[0]
    probFile = open(fileName, 'rU')
    testTripletsFile = open('testTriplets.txt', 'rU')

    probs = []
    for line in probFile:
        probs.append(float(line.split()[0]))

    # print probs

    # Test file reading
    testRow = []; testCol = []; testDat = []
    for line in testTripletsFile:
        arr = line.split()
        testRow.append(int(float(arr[0])))
        testCol.append(int(float(arr[1])))
        testDat.append(int(float(arr[2])))

    j = 0
    while j < len(probs):
        if probs[j] > 0.0005:
            x = probs.pop(j)
            print x
            testDat.pop(j)
            continue
        else:
            j = j + 1

    print "yo"

    for element in probs:
        if element > 0.0001:
            print "shit"

    fpr, tpr = ROC(testDat, probs)
    print "FPR: " + str(fpr)
    print "TPR: " + str(tpr)

    precision, recall, thresholds = pr(testDat, probs)
    print "precision: "
    for i in range(len(precision)):
        print str(i) + ": " + str(precision[i])

    print "recall: "
    for i in range(len(recall)):
        print str(i) + ": " + str(recall[i])

    print "thresholds: "
    for i in range(len(thresholds)):
        print str(i) + ": " + str(thresholds[i])

    #Manually calculating the recall:
    # thres =  0.0385712031865
    # fn = 0
    # tp = 0
    # fp = 0
    # tn = 0
    # for i in range(len(probs)):
    #     if probs[i] < thres:
    #         if testDat[i] == 1:
    #             fn = fn + 1
    #     if probs[i] > thres:
    #         if testDat[i] == 1:
    #             tp = tp + 1
    #     if probs[i] > thres:
    #         if testDat[i] == 0:
    #             fp = fp + 1
    #     if probs[i] < thres:
    #         if testDat[i] == 0:
    #             tn = tn + 1
    # print "tp: " + str(tp)
    # print "fp: " + str(fp)
    # print "fn: " + str(fn)
    # print "tn: " + str(tn)

    zeroProb = []
    oneProb = []

    for i in range(len(testDat)):
        if testDat[i] == 0:
            zeroProb.append(probs[i])
        else:
            oneProb.append(probs[i])

    groups = range(2)
    a = np.array(zeroProb)
    b = np.array(oneProb)
    data = []
    
    data.append(a)
    data.append(b)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    violin_plot(ax,data,groups,bp=0)
    pl.show()


if __name__ == "__main__":
    main(sys.argv[1:])