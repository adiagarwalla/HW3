import numpy as np 
import sys 

def main(args):
    fileName = args[0]
    dictionaryFile = open('addrIndex.txt', 'rU')
    addressesToConvert = open(fileName, 'rU')

    mapping = {}
    for line in dictionaryFile:
        arr = line.split()
        mapping[int(arr[0])] = int(arr[1])

    # for line in addressesToConvert:
    #     arr = line.split()
    #     if arr[0] in mapping:
    #         print arr[0] + ": " + str(mapping[int(arr[0])])

    counter = 0
    for line in addressesToConvert:
        arr = line.split()
        if int(arr[0]) in mapping:
            counter = counter + 1

    print counter

if __name__ == "__main__":
    main(sys.argv[1:])