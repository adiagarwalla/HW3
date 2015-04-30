import sys

def main(args):
    fileName = args[0]
    blockChain = open(fileName, 'rU')
    clusterSets = []
    counter = 0

    for line in blockChain:
#        print "-------"
        if counter == 0:
            print counter
            counter = counter + 1
            prev = line.split()[1]
            prevAdd = line.split()[2]

        else:
            # print prevAdd
            # print prev
            print counter
            arr = line.split()

            if arr[1] == prev:
                clusterIndexPrev = -1
                clusterIndexNow = -1

                #Finding the cluster index that each address belongs to
                for i in range(len(clusterSets)):
#                    print arr[2]
#                    print prevAdd
#                    print clusterSets[i]
                    if str(prevAdd) in clusterSets[i]:
                        clusterIndexPrev = i
                    if str(arr[2]) in clusterSets[i]:
                        clusterIndexNow = i
                    if clusterIndexPrev != -1 and clusterIndexNow != -1:
                        break

#                print clusterIndexPrev
#                print clusterIndexNow
                #If they both belong to a cluster then make them into a union
                if (clusterIndexPrev != -1) and (clusterIndexNow != -1) and clusterIndexNow != clusterIndexPrev:
#                    print "1--"
                    clusterSets[clusterIndexPrev] = clusterSets[clusterIndexPrev].union(clusterSets[clusterIndexNow])
                    clusterSets.remove(clusterSets[clusterIndexNow])

                #If one belongs to cluster and the other one doesn't
                if (clusterIndexPrev != -1) and (clusterIndexNow == -1):
#                    print "2--"
                    clusterSets[clusterIndexPrev].add(arr[2])

                if (clusterIndexPrev == -1) and (clusterIndexNow != -1):
#                    print "3--"
                    clusterSets[clusterIndexNow].add(prevAdd)
                
                #If neither index belongs to cluster then make a new cluster
                if (clusterIndexPrev == -1) and (clusterIndexNow == -1):
#                    print "4--"
                    setNew = set()
                    setNew.add(prevAdd)
                    setNew.add(arr[2])
                    clusterSets.append(setNew)

            prevAdd = arr[2]
            prev = arr[1]
            counter = counter + 1

    print "Summary statistics:"
    print "Total number of cluster sets: " + str(len(clusterSets))
    #We want to find the top 10 largest clusters and their sizes
    sizeDictionary = {}
    for clusterSet in clusterSets:
        if len(clusterSet) not in sizeDictionary:
            sizeDictionary[len(clusterSet)] = []
            sizeDictionary[len(clusterSet)].append(clusterSet)
        else:
            sizeDictionary[len(clusterSet)].append(clusterSet)

    import operator
    sorted_x = sorted(sizeDictionary.items(), key=operator.itemgetter(0), reverse=True)
    print "the largest cluster size = " + str(sorted_x[0][0])
    print "The largest size sets:"
    print sorted_x


    print 
    print "--------------------------------------"
    print

    print "Actual data matrix"
    print clusterSets 


if __name__ == "__main__":
    main(sys.argv[1:])