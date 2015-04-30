import sys
import operator

def main(args):
    fileName = args[0]
    dataFile = open(fileName, 'rU')

    freq = {}
    freqGiver = {}
    length = 444075

    for i in range(length):
        freq[i] = 0
        freqGiver[i] = 0
    
    for line in dataFile:
        arr = line.split()
        freq[int(float(arr[1]))] = freq[int(float(arr[1]))] + 1
        freqGiver[int(float(arr[0]))] = freqGiver[int(float(arr[0]))] + 1

    sorted_freq = sorted(freq.items(), key=operator.itemgetter(1))
    sorted_freqGiver = sorted(freqGiver.items(), key=operator.itemgetter(1))
    #sorted_dict = dict(sorted_freq)
    for i in range(length, length - 500, -1): 
        print sorted_freq[i - 1]

    print "----------------------"

    for i in range(length, length - 50, -1): 
        print sorted_freqGiver[i - 1]

if __name__ == "__main__":
    main(sys.argv[1:])