__author__ = 'goe'
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