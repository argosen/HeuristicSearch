__author__ = 'goe'

import LocalSearch as ls
import dataModel as dm
from Neighbors import NeighborType as neigh_type
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab
import time

def showElapsedTimeBars( dataList ):
    # Example data
    people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
    repetitionList = [str(i) for i in range(10, 110, 10)]
    y_pos = np.arange(len(repetitionList))
    performance = [data.getAverageTime() for data in dataList]

    plt.bar(y_pos, performance, align='center', alpha=0.4)
    plt.xticks(y_pos, repetitionList)
    plt.ylabel('Elapsed Time')
    plt.title('Iteration mean elapsed time')

    plt.show()

def showElapsedTimeLine( dataListA, dataListB ):
    # Example data
    repetitionList = [str(i) for i in range(10, 110, 10)]
    y_pos = np.arange(len(repetitionList))

    performanceA = [data.getAverageTime() for data in dataListA]
    performanceB = [data.getAverageTime() for data in dataListB]

    plt.plot(y_pos, performanceA, 'r', label='Genetic Algorithm')
    plt.plot(y_pos, performanceB, 'b', label='Local Search')
    plt.xticks(y_pos, repetitionList)
    plt.ylabel('Elapsed Time')
    plt.title('Iteration mean elapsed time')

    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')

    plt.show()

def showMeanErrorLine( dataListGA, dataListLS, bestKnownSolutionList ):
    n_groups = len(dataListGA)
    # Compute error stdev and mean
    error_mean_ga = np.array([])
    error_stdev_ga = np.array([])
    for i, item in enumerate(dataListGA):
        errors = item.computation_values - bestKnownSolutionList[i]
        error_mean_ga = np.append(error_mean_ga, errors.mean())
        error_stdev_ga = np.append(error_stdev_ga, np.sqrt(errors.var()))

    error_mean_ls = np.array([])
    error_stdev_ls = np.array([])
    for i, item in enumerate(dataListLS):
        errors = item.computation_values - bestKnownSolutionList[i]
        error_mean_ls = np.append(error_mean_ls, errors.mean())
        error_stdev_ls = np.append(error_stdev_ls, np.sqrt(errors.var()))

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, error_mean_ls, bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=error_stdev_ls,
                     error_kw=error_config,
                     label='Local Search')

    rects2 = plt.bar(index + bar_width, error_mean_ga, bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=error_stdev_ga,
                     error_kw=error_config,
                     label='Genetic Algorithm')

    plt.xlabel('Factory amount')
    plt.ylabel('Error')
    plt.title('Error for each file and algorithm (Using mean values)')
    plt.xticks(index + bar_width, [str(n) for n in range(10, 110, 10)])
    plt.legend(loc='upper left', shadow=True)

    plt.tight_layout()
    plt.show()


def showMinErrorLine( dataListGA, dataListLS, bestKnownSolutionList ):
    n_groups = len(dataListGA)
    # Compute error stdev and mean
    error_min_ga = np.array([])
    for i, item in enumerate(dataListGA):
        errors = item.computation_values - bestKnownSolutionList[i]
        error_min_ga = np.append(error_min_ga, errors.min())

    error_min_ls = np.array([])
    for i, item in enumerate(dataListLS):
        errors = item.computation_values - bestKnownSolutionList[i]
        error_min_ls = np.append(error_min_ls, errors.min())

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, error_min_ls, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Local Search')

    rects2 = plt.bar(index + bar_width, error_min_ga, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Genetic Algorithm')

    plt.xlabel('Factory amount')
    plt.ylabel('Error')
    plt.title('Error for each file and algorithm (Using best values)')
    plt.xticks(index + bar_width, [str(n) for n in range(10, 110, 10)])
    plt.legend(loc='upper left', shadow=True)

    plt.tight_layout()
    plt.show()

bestKnownSolution = [3971134, 17412718, 95424438, 150786632, 254375088, 442535568, 661277026, 852010960, 1162218930, 1431932318]
fileList = ["Cebe.qap.n10.1", "Cebe.qap.n20.1", "Cebe.qap.n30.1", "Cebe.qap.n40.1", "Cebe.qap.n50.1", "Cebe.qap.n60.1", "Cebe.qap.n70.1", "Cebe.qap.n80.1", "Cebe.qap.n90.1", "Cebe.qap.n100.1",]
#fileList = ["Cebe.qap.n10.1"]
lsConfigurations = [neigh_type.TwoOpt, neigh_type.Swap]
lsConfigLabels = ['TwoOpt', 'Swap']

repetition = 15
fixedSearchRange = 15

plotResults = True
needToCompute = True

gaDataList = np.array([])
lsDataList = np.array([])

# Local search computation
for k, currentFile in enumerate(fileList):
    maxValue = 0
    minValueList = np.array([])
    colorList = np.array([])

    gaData = dm.ResultDataModel("./solutions/"+currentFile+"_ga_solutions")
    gaData.loadDataFile()
    gaDataList = np.append(gaDataList, gaData)

    lsData = dm.ResultDataModel("./solutions/"+currentFile+"_ls_solutions")
    lsData.loadDataFile()
    lsDataList = np.append(lsDataList, lsData)

print("Loaded " + str(len(gaDataList)) + " files")


for ls, ga, best in zip(lsDataList, gaDataList, bestKnownSolution):
    print('|\tLS Values (Error)\t\t|\tGA Values (Error)\t\t|\tBest known value\t\t|')
    for lsVal, gaVal in zip(ls.computation_values, ga.computation_values):
        print('|\t' + str(lsVal) + ' (' + str(lsVal-best) + ')\t|\t' + str(gaVal) + ' (' + str(gaVal-best) + ')\t|\t' + str(best) + '\t|')
    print('|----------------------------------------------------------|')

# error
showMinErrorLine( gaDataList, lsDataList, bestKnownSolution )
showMeanErrorLine( gaDataList, lsDataList, bestKnownSolution )


# time
showElapsedTimeLine(gaDataList, lsDataList)
#showElapsedTimeBars(gaDataList)