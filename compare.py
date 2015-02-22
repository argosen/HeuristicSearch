__author__ = 'goe'

import LocalSearch as ls
import dataModel as dm
from Neighbors import NeighborType as neigh_type
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab
import time

def runLocalSearchAlgorithm(currentFile, config, repetitions, searchRange):
    localSearch = ls.LocalSearch("./Instances/"+currentFile)

    computation_values = np.array([])
    computation_solutions = np.array([[]])
    computation_evaluations = np.array([])
    computation_time = np.array([])

    print("Computing file '" + currentFile + "'")
    for iteration in range(0, repetitions):
        start_time = time.time()
        # Run the local search once
        [best_val, best_sol, best_evaluations] = localSearch.solve(searchRange, config) #neigh_type.Swap

        elapsed_time = time.time() - start_time

        # History
        # Store the result
        computation_time = np.append(computation_time, elapsed_time)
        computation_values = np.append(computation_values, best_val)
        computation_evaluations = np.append(computation_evaluations, best_evaluations)
        if iteration == 0:
            computation_solutions = np.array([best_sol])
        else:
            computation_solutions = np.append(computation_solutions, [best_sol], axis=0)

        print("\t[Iteration #" + str(iteration) + "] -> " + str(best_val) + " | " + str(best_sol) + " | " + str(best_evaluations))


    # Get best value
    best_value_index = np.where(computation_values == computation_values.min())
    print("Best Value: " + str(computation_values[best_value_index]))
    print("Best Solution: " + str(computation_solutions[best_value_index]))

    # Compute mean and variance
    print("Execution mean = " + str(computation_values.mean()))
    print("Execution variance = " + str(computation_values.var()))

    # Store results in a file
    outputModel = dm.ResultDataModel("./solutions_test_2/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(repetitions)+"_s"+str(searchRange))
    outputModel.setValueList(computation_values)
    outputModel.setSolutionList(computation_solutions)
    outputModel.setEvaluationNumberList(computation_evaluations)
    outputModel.setTimeList(computation_time)
    outputModel.saveDataFile()

    return computation_values[best_value_index]

bestKnownSolution = [3971134, 17412718, 95424438, 150786632, 254375088, 442535568, 661277026, 852010960, 1162218930, 1431932318]
#fileList = ["Cebe.qap.n10.1", "Cebe.qap.n20.1", "Cebe.qap.n30.1", "Cebe.qap.n40.1", "Cebe.qap.n50.1", "Cebe.qap.n60.1", "Cebe.qap.n70.1", "Cebe.qap.n80.1", "Cebe.qap.n90.1", "Cebe.qap.n100.1",]
fileList = ["Cebe.qap.n10.1"]
lsConfigurations = [neigh_type.TwoOpt, neigh_type.Swap]
lsConfigLabels = ['TwoOpt', 'Swap']

repetition = 15
fixedSearchRange = 15

plotResults = True
needToCompute = True

# Local search computation
for k, currentFile in enumerate(fileList):
    maxValue = 0
    minValueList = np.array([])
    colorList = np.array([])
    for i, config in enumerate(lsConfigurations):
        bestResultList = np.array([])
        for iteration in range(0, 30):
            bestResult = runLocalSearchAlgorithm(currentFile, config, repetition, fixedSearchRange)
            bestResultList = np.append(bestResultList, bestResult)

        lsData = dm.ResultDataModel("./solutions_test_2/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(repetition)+"_s"+str(fixedSearchRange))
        lsData.loadDataFile()

        mean = bestResultList.mean()
        stdev = np.sqrt(bestResultList.var())
        rangeValue = 3 * stdev
        rn = np.linspace(mean-rangeValue, mean+rangeValue, 100)
        # Create plots with pre-defined labels.
        curve = mlab.normpdf(rn, mean, stdev)
        p = plt.plot(rn, curve, label=str(config))

        minValueList = np.append(minValueList, lsData.getBestValue())
        colorList = np.append(colorList, p[0]._color)

        if maxValue < max(curve):
            maxValue = max(curve)

    plt.plot([bestKnownSolution[k], bestKnownSolution[k]], [0, maxValue], 'k:', label='Best Known Solution')

    for k, minVal in enumerate(minValueList):
        plt.plot([minVal, minVal], [0, maxValue], colorList[k])

    plt.title('LS repetition comparatives', fontsize=16)
    legend = plt.legend(loc='upper right', shadow=True, fontsize='12')
    pylab.savefig('./solutions_test_2/LS_neighbor_comparatives_gaussian.png', bbox_inches='tight')
    if plotResults:
        plt.show()
    else:
        plt.clf()