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

def main(args):
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    pr(y_true, y_scores)

if __name__ == "__main__":
    main(sys.argv[1:])