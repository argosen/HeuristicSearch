__author__ = 'goe'
from pylab import *
import numpy as np

class DataModel:

    def __init__(self):

        self.filename = ""
        self.numberOfElements = 0
        self.distanceMatrix = 0
        self.flowMatrix = 0

    def __init__(self, filename):

        self.filename = ""
        self.numberOfElements = 0
        self.distanceMatrix = 0
        self.flowMatrix = 0
        self.loadDataFile(filename)


    def loadDataFile(self, filename):
        # Clean the previous data
        self.filename = filename
        self.numberOfElements = 0

        infile = open(filename, 'r')
        # set the initial state
        state = "none"

        firstLine = True

        cont = 0
        for line in infile:
            # ignore comented lines
            if(line.startswith('#')):
                continue

            # Conditional state changes
            if state == "none" :
                state = "readNumber"
            elif state == "readDistanceMatrix" and cont == self.numberOfElements:
                state = "readFlowMatrix"
                firstLine = True

            # store the material file name
            if state == "readNumber":
                self.numberOfElements = int(line)
                state = "readDistanceMatrix"
                firstLine = True
                continue

            # Start reading the distance matrix
            if state == "readDistanceMatrix":
                if firstLine:
                    distLine = line
                    firstLine = False
                else:
                    distLine = distLine + ';' + line
                cont = cont + 1

            # Start reading the distance matrix
            if state == "readFlowMatrix":
                if firstLine:
                    flowLine = line
                    firstLine = False
                else:
                    flowLine = flowLine + ';' + line
                cont = cont + 1

        self.distanceMatrix = np.matrix(distLine)
        self.flowMatrix = np.matrix(flowLine)

        return True

    def showMatrices(self):
        print(self.numberOfElements)
        print(self.distanceMatrix)
        print(self.flowMatrix)




class ResultDataModel:

    # computation_values = np.array([])
    # computation_solutions = np.array([[]])
    # computation_evaluations = np.array([])
    def __init__(self, filename):
        self.filename = filename
        self.computation_values = np.array([])
        self.computation_solutions = np.array([[]])
        self.computation_evaluations = np.array([])
        self.computation_time = np.array([])

    def setValueList(self, valueList):
        self.computation_values = valueList

    def setSolutionList(self, solutionList):
        self.computation_solutions = solutionList

    def setEvaluationNumberList(self, evalutationNumberList):
        self.computation_evaluations = evalutationNumberList

    def setTimeList(self, timeList):
        self.computation_time = timeList

    def loadDataFile(self):
        # load data file
        # store block sizes
        #self.computation_values = genfromtxt(self.filename + "_values.txt")
        #self.computation_solutions = genfromtxt(self.filename + "_solutions.txt")
        #self.computation_evaluations = genfromtxt(self.filename + "_evaluations.txt")

        f = file(self.filename + ".npy","rb")

        self.computation_values = np.load(f)
        self.computation_solutions = np.load(f)
        self.computation_evaluations = np.load(f)
        self.computation_time = np.load(f)

        f.close()

        print("Data loaded from : " + self.filename + ".npy")

    def saveDataFile(self):
        # save data file
        # store block sizes
        #savetxt(self.filename + "_values.txt", self.computation_values)
        #savetxt(self.filename + "_solutions.txt", self.computation_solutions)
        #savetxt(self.filename + "_evaluations.txt", self.computation_evaluations)

        f = file(self.filename + ".npy","wb")

        np.save(f, self.computation_values)
        np.save(f, self.computation_solutions)
        np.save(f, self.computation_evaluations)
        np.save(f, self.computation_time)

        f.close()

        print("Data saved in : " + self.filename)

    def plotResultValues(self):
        axes([0.1, 0.15, 0.8, 0.75])
        plot(self.computation_values.tolist())

        #horizontalLineInValue = 5
        #plot([0,10],[horizontalLineInValue,horizontalLineInValue])

        title('Evaluation results', fontsize=20)

        show()

    def getValueMean(self):
        return self.computation_values.mean()

    def getValueVariance(self):
        return self.computation_values.var()

    def getValueStdev(self):
        return math.sqrt(self.computation_values.var())

    def getBestValue(self):
        best_value_index = np.where(self.computation_values == self.computation_values.min())[0]
        if len(best_value_index) == 1:
            return self.computation_values[best_value_index]
        else:
            return self.computation_values[best_value_index[0]]

    def getBestSolution(self):
        best_value_index = np.where(self.computation_values == self.computation_values.min())[0]
        if len(best_value_index) == 1:
            return self.computation_solutions[best_value_index]
        else:
            return self.computation_solutions[best_value_index[0]]

    def getAverageTime(self):
        return sum(self.computation_time)/len(self.computation_time)