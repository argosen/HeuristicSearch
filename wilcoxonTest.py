__author__ = 'goe'

import numpy as np
import dataModel as dm

def wilcoxonTest(aResultList, bResultList, plotResult=True):
    # compute diff
    for i in range(len(gaResults)):
        diff = np.append(diff, bResultList[i]-aResultList[i])
    absDiff = abs(diff)

    # rank results
    tableSize = len(diff)
    rank = np.zeros((1, tableSize))[0]
    valuesToBeRanked = tableSize
    maxValue = absDiff.max()
    while valuesToBeRanked > 0:
        # get best value index/indices
        best_value_index = np.where(absDiff == absDiff.min())[0]
        currentValueToRank = 0
        # Compute rank value for the detected indices
        for i in range(len(best_value_index)):
            currentValueToRank += tableSize - valuesToBeRanked + 1
            valuesToBeRanked -= 1
        currentValueToRank = float(currentValueToRank) / len(best_value_index)
        # put the rank values in the places
        rank[best_value_index] = currentValueToRank
        absDiff[best_value_index] = maxValue+1

    # get positives and negatives
    r_plus = 0
    r_minus = 0
    for i in range(tableSize):
        if diff[i] <= 0:
            r_minus += diff[i]

        if diff[i] >= 0:
            r_plus += diff[i]

    # get min
    T = min(r_plus, r_minus)
    N = tableSize
    z = (T - (1/4) * N * (N+1)) / np.sqrt((1/24) * N * (N+1) * (2*N+1))

    # print table
    if plotResult:
        print('| \tLS Res\t | \tGA Res\t | \tdiff\t| \tRank\t |')
        for i in range(tableSize):
            print('| \t'+str(aResultList[i])+'\t | \t'+str(bResultList[i])+'\t | \t'+str(diff[i])+'\t | \t'+str(rank[i])+'\t |')

        print('-------------------------')
        print('R+ = '+str(r_plus))
        print('R- = '+str(r_minus))
        print('T = '+str(T))

    return z


fileList = ["Cebe.qap.n10.1", "Cebe.qap.n20.1", "Cebe.qap.n30.1", "Cebe.qap.n40.1", "Cebe.qap.n50.1", "Cebe.qap.n60.1", "Cebe.qap.n70.1", "Cebe.qap.n80.1", "Cebe.qap.n90.1", "Cebe.qap.n100.1",]
lsResults = np.array([])
gaResults = np.array([])
diff = np.array([])

# Generate table
simulate = True
if not simulate:
    for currentFile in fileList:
        gaModel = dm.ResultDataModel("./solutions/"+currentFile+"_ga_solutions")
        lsModel = dm.ResultDataModel("./solutions/"+currentFile+"_ls_solutions")
        gaResults = np.append(gaResults, gaModel.getValueMean())
        lsResults = np.append(lsResults, lsModel.getValueMean())
else:
    gaResults = np.array([10, 21, 23, 32, 12, 13, 14, 12, 43, 23])
    lsResults = np.array([3, 1, 50, 3, 24, 42, 25, 6, 6, 14])

z = wilcoxonTest(lsResults, gaResults)
