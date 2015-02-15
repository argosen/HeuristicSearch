__author__ = 'goe'

import LocalSearch as ls
import GeneticAlgorithm as ga
from Neighbors import NeighborType as neigh_type
import numpy as np

import dataModel as dm

#model = dm.DataModel()
#model.loadDataFile("test.txt")
#model.showMatrices()

# localSearch = ls.LocalSearch("test.txt")
# fileList = ["Cebe.qap.n10.1","Cebe.qap.n20.1","Cebe.qap.n30.1","Cebe.qap.n40.1","Cebe.qap.n50.1","Cebe.qap.n60.1","Cebe.qap.n70.1","Cebe.qap.n80.1","Cebe.qap.n90.1","Cebe.qap.n100.1",]
fileList = ["Cebe.qap.n10.1"]
searchRange = 10 # Multi-start parameter
repetitions = 10 # times to compute mean and variance

# Poblational algorithm computation
for currentFile in fileList:
    geneticAlgorithm = ga.GeneticAlgorithm("./Instances/"+currentFile)

    computation_values = np.array([])
    computation_solutions = np.array([[]])
    computation_evaluations = np.array([])

    print("Computing file '" + currentFile + "'")
    for iteration in range(0, repetitions):
        # Run the local search once
        [best_val, best_sol, best_evaluations] = geneticAlgorithm.solve(searchRange)

        # History
        # Store the result
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
    outputModel = dm.ResultDataModel("testOutputFile")
    outputModel.setValueList(computation_values)
    outputModel.setSolutionList(computation_solutions)
    outputModel.setEvaluationNumberList(computation_evaluations)
    outputModel.saveDataFile()

exit()
# Local search computation
for currentFile in fileList:
    localSearch = ls.LocalSearch("./Instances/"+currentFile)

    computation_values = np.array([])
    computation_solutions = np.array([[]])
    computation_evaluations = np.array([])

    print("Computing file '" + currentFile + "'")
    for iteration in range(0, repetitions):
        # Run the local search once
        [best_val, best_sol, best_evaluations] = localSearch.solve(searchRange, neigh_type.Swap)

        # History
        # Store the result
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

