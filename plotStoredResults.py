__author__ = 'goe'

import dataModel as dm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from Neighbors import NeighborType as neigh_type

# fileList = ["Cebe.qap.n10.1","Cebe.qap.n20.1","Cebe.qap.n30.1","Cebe.qap.n40.1","Cebe.qap.n50.1","Cebe.qap.n60.1","Cebe.qap.n70.1","Cebe.qap.n80.1","Cebe.qap.n90.1","Cebe.qap.n100.1",]
fileList = ["Cebe.qap.n10.1"]
gaConfigurations = [1, 2, 3]
gaConfigLabels = ['Preserve first half', 'Preserve First/Last half', 'Interchange one']
lsConfigurations = [neigh_type.Swap, neigh_type.TwoOpt]
lsConfigLabels = ['Swap', 'TwoOpt']
bestKnownSolution = [3971134, 17412718, 95424438, 150786632, 254375088, 442535568, 661277026, 852010960, 1162218930, 1431932318]

def mainFunction():
    solutionFolder = "./solutions/"
    for i, currentFile in enumerate(fileList):
        showGaMateComparativeGaussian(solutionFolder+currentFile, gaConfigurations, gaConfigLabels, bestKnownSolution[i])


        showSolutionComparative(solutionFolder+currentFile, ga_config_id=gaConfigurations[i], ls_config_id=lsConfigurations[i], bestSolution=bestKnownSolution[i])


        showGaMateComparative(solutionFolder+currentFile, gaConfigurations, gaConfigLabels)
        showLsNeighComparative(solutionFolder+currentFile, lsConfigurations, lsConfigLabels)


def showGaMateComparative(filename, gaConfig, configLabels):
    for i, config in enumerate(gaConfig):
        gaData = dm.ResultDataModel(filename+"_ga_solutions_"+str(config))
        gaData.loadDataFile()

        dataSize = len(gaData.computation_values)
        rn = range(0, dataSize)
        # Create plots with pre-defined labels.
        plt.plot(rn, gaData.computation_values, label=configLabels[i])

    plt.title('GA Mate function Comparatives', fontsize=20)
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()


def showGaMateComparativeGaussian(filename, gaConfig, configLabels, bestSolution=0):
    maxValue = 0
    for i, config in enumerate(gaConfig):
        gaData = dm.ResultDataModel(filename+"_ga_solutions_"+str(config))
        gaData.loadDataFile()

        mean = gaData.getValueMean()
        stdev = gaData.getValueStdev()
        rangeValue = 3 * stdev
        rn = np.linspace(mean-rangeValue, mean+rangeValue, 100)
        # Create plots with pre-defined labels.
        curve = mlab.normpdf(rn, mean, stdev)
        plt.plot(rn, curve, label=configLabels[i])

        if maxValue < max(curve):
            maxValue = max(curve)

    if bestSolution > 0:
        plt.plot([bestSolution, bestSolution], [0, maxValue], 'k:', label='Best Known Solution')

    plt.title('GA Mate function Comparatives', fontsize=20)
    legend = plt.legend(loc='upper right', shadow=True, fontsize='12')
    plt.show()

def showLsNeighComparative(filename, lsConfig, configLabels):
    for i, config in enumerate(lsConfig):
        lsData = dm.ResultDataModel(filename+"_ls_solutions_"+str(config))
        lsData.loadDataFile()

        dataSize = len(lsData.computation_values)
        rn = range(0, dataSize)
        # Create plots with pre-defined labels.
        plt.plot(rn, lsData.computation_values, label=configLabels[i])

    plt.title('LS Neighbor function Comparatives', fontsize=20)
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()

def showSolutionComparative(fileName, ga_config_id=3, ls_config_id=neigh_type.Swap, bestSolution=0):
    gaData = dm.ResultDataModel(fileName+"_ga_solutions_"+str(ga_config_id))
    gaData.loadDataFile()

    lsData = dm.ResultDataModel(fileName+"_ls_solutions_"+str(ls_config_id))
    lsData.loadDataFile()

    dataSize = len(gaData.computation_values)
    rn = range(0, dataSize)
    # Create plots with pre-defined labels.
    plt.plot(rn, gaData.computation_values, label='Genetic Algorithm')
    plt.plot(rn, lsData.computation_values, label='Local Search')

    if bestSolution > 0:
        plt.plot([0, dataSize], [bestSolution, bestSolution], 'k:', label='Best Known Solution')

    plt.title('LS vs GA Comparative', fontsize=20)
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')

    plt.show()



mainFunction()