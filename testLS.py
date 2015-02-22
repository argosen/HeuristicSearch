__author__ = 'goe'

import LocalSearch as ls
from Neighbors import NeighborType as neigh_type
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab

import dataModel as dm

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
    outputModel = dm.ResultDataModel("./solutions_test/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(repetitions)+"_s"+str(searchRange))
    outputModel.setValueList(computation_values)
    outputModel.setSolutionList(computation_solutions)
    outputModel.setEvaluationNumberList(computation_evaluations)
    outputModel.setTimeList(computation_time)
    outputModel.saveDataFile()


# localSearch = ls.LocalSearch("test.txt")
#fileList = ["Cebe.qap.n10.1", "Cebe.qap.n20.1", "Cebe.qap.n30.1", "Cebe.qap.n40.1", "Cebe.qap.n50.1", "Cebe.qap.n60.1", "Cebe.qap.n70.1", "Cebe.qap.n80.1", "Cebe.qap.n90.1", "Cebe.qap.n100.1",]
fileList = ["Cebe.qap.n10.1"]
lsConfigurations = [neigh_type.TwoOpt, neigh_type.Swap]
lsConfigLabels = ['TwoOpt', 'Swap']

fig, ax = plt.subplots()

searchRangeList = [10, 20, 30, 40] # Multi-start parameter
repetitionList = [10, 15, 20, 25]

fixedSearchRange = 20
fixedRepetitions = 20

bestKnown = 3971134

plotResults = False
needToCompute = True

if True:
    # Local search computation
    for currentFile in fileList:
        for i, config in enumerate(lsConfigurations):
            # generate data files
            for repetition in repetitionList:
                if needToCompute:
                    runLocalSearchAlgorithm(currentFile, config, repetition, fixedSearchRange)
            # load and show data files
            # show results
            for j, repetition in enumerate(repetitionList):
                lsData = dm.ResultDataModel("./solutions_test/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(repetition)+"_s"+str(fixedSearchRange))

                lsData.loadDataFile()

                dataSize = len(lsData.computation_values)
                rn = range(0, dataSize)
                # load to plot
                plt.plot(rn, lsData.computation_values, label=str(repetitionList[j])+" repetitions")


            plt.plot([0, dataSize], [bestKnown, bestKnown], 'k:', label='Best Known Solution')
            # plot data
            plt.title('LS repetition comparatives ('+str(lsConfigurations[i])+')', fontsize=20)
            legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
            pylab.savefig('./solutions_test/LS_repetition_comparatives_All_'+str(lsConfigurations[i])+'.png', bbox_inches='tight')
            if plotResults:
                plt.show()
            else:
                plt.clf()

            #show gaussian
            maxValue = 0
            minValueList = np.array([])
            colorList = np.array([])
            for j, repetition in enumerate(repetitionList):
                lsData = dm.ResultDataModel("./solutions_test/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(repetition)+"_s"+str(fixedSearchRange))
                lsData.loadDataFile()

                mean = lsData.getValueMean()
                stdev = lsData.getValueStdev()
                rangeValue = 3 * stdev
                rn = np.linspace(mean-rangeValue, mean+rangeValue, 100)
                # Create plots with pre-defined labels.
                curve = mlab.normpdf(rn, mean, stdev)
                p = plt.plot(rn, curve, label=str(repetitionList[j])+" repetitions")

                minValueList = np.append(minValueList, lsData.getBestValue())
                colorList = np.append(colorList, p[0]._color)

                if maxValue < max(curve):
                    maxValue = max(curve)

            plt.plot([bestKnown, bestKnown], [0, maxValue], 'k:', label='Best Known Solution')

            for k, minVal in enumerate(minValueList):
                plt.plot([minVal, minVal], [0, maxValue], colorList[k])

            plt.title('LS repetition comparatives ('+str(lsConfigurations[i])+')', fontsize=20)
            legend = plt.legend(loc='upper right', shadow=True, fontsize='12')
            pylab.savefig('./solutions_test/LS_repetition_comparatives_gaussians'+str(lsConfigurations[i])+'.png', bbox_inches='tight')
            if plotResults:
                plt.show()
            else:
                plt.clf()



        # show elapsed time avg
        finalAvg = np.array([])
        for i, config in enumerate(lsConfigurations):
            # generate data files
            avg = 0
            for repetition in repetitionList:
                lsData = dm.ResultDataModel("./solutions_test/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(repetition)+"_s"+str(fixedSearchRange))
                lsData.loadDataFile()
                avg += lsData.getAverageTime()

            finalAvg = np.append(finalAvg, avg / len(repetitionList))

        ind = np.arange(len(lsConfigurations))  # the x locations for the groups

        plt.barh(ind, finalAvg, xerr=0, align='center', alpha=0.4)
        plt.yticks(ind, lsConfigLabels)
        plt.xlabel('Time')
        plt.title('LS Elapsed time')

        pylab.savefig('./solutions_test/LS_elapsed_time_comparatives_repetitions_'+str(lsConfigurations[i])+'.png', bbox_inches='tight')
        if plotResults:
            plt.show()
        else:
            plt.clf()










needToCompute = True
# Local search computation
for currentFile in fileList:
    for i, config in enumerate(lsConfigurations):
        # generate data files
        for start in searchRangeList:
            if needToCompute:
                runLocalSearchAlgorithm(currentFile, config, fixedRepetitions, start)
        # load and show data files
        # show results
        for j, start in enumerate(searchRangeList):
            lsData = dm.ResultDataModel("./solutions_test/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(fixedRepetitions)+"_s"+str(start))

            lsData.loadDataFile()

            dataSize = len(lsData.computation_values)
            rn = range(0, dataSize)
            # load to plot
            plt.plot(rn, lsData.computation_values, label=str(searchRangeList[j])+" starts")


        plt.plot([0, dataSize], [bestKnown, bestKnown], 'k:', label='Best Known Solution')
        # plot data
        plt.title('LS start comparatives ('+str(lsConfigurations[i])+')', fontsize=20)
        legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')

        pylab.savefig('./solutions_test/LS_start_comparatives_All_'+str(lsConfigurations[i])+'.png', bbox_inches='tight')
        if plotResults:
            plt.show()
        else:
            plt.clf()

        #show gaussian
        maxValue = 0
        minValueList = np.array([])
        colorList = np.array([])
        for j, start in enumerate(searchRangeList):
            lsData = dm.ResultDataModel("./solutions_test/"+currentFile+"_ls_solutions_"+str(config)+"_r"+str(fixedRepetitions)+"_s"+str(start))
            lsData.loadDataFile()

            mean = lsData.getValueMean()
            stdev = lsData.getValueStdev()
            rangeValue = 3 * stdev
            rn = np.linspace(mean-rangeValue, mean+rangeValue, 100)
            # Create plots with pre-defined labels.
            curve = mlab.normpdf(rn, mean, stdev)
            p = plt.plot(rn, curve, label=str(searchRangeList[j])+" starts")

            minValueList = np.append(minValueList, lsData.getBestValue())
            colorList = np.append(colorList, p[0]._color)

            if maxValue < max(curve):
                maxValue = max(curve)

        printMaxValue = maxValue + (maxValue*0.1)
        plt.plot([bestKnown, bestKnown], [0, printMaxValue], 'k:', label='Best Known Solution')

        for k, minVal in enumerate(minValueList):
            plt.plot([minVal, minVal], [0, maxValue], colorList[k])

        plt.title('LS start comparatives ('+str(lsConfigurations[i])+')', fontsize=20)
        legend = plt.legend(loc='upper right', shadow=True, fontsize='12')
        pylab.savefig('./solutions_test/LS_start_elapsed_time_comparatives_gaussians_'+str(lsConfigurations[i])+'.png', bbox_inches='tight')
        if plotResults:
            plt.show()
        else:
            plt.clf()
