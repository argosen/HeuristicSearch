__author__ = 'goe'

import dataModel as dm
import numpy as np
import Neighbors as neigh
from Neighbors import NeighborType as neighType


class LocalSearch:

    def __init__(self, filename):
        self.model = dm.DataModel(filename)
        self.result = np.array(range(self.model.numberOfElements))
        self.init_sol = np.random.permutation(self.model.numberOfElements)

    def costFunction(self, perm):
        res = 0
        n = self.model.numberOfElements

        for i in range(n):
            for j in range(n):
                sig_i = perm[i]
                sig_j = perm[j]
                res = res + self.model.distanceMatrix[sig_i, sig_j] * self.model.flowMatrix[i, j]

        return res

    def setInitialState(self):
        self.init_sol = np.random.permutation(self.model.numberOfElements)

    def solve(self, n_starts=100):
        [best_val, best_sol, best_evaluations] = self.runLocalSearchFast()#self.runLocalSearch(neigh_type)
        for i in range(n_starts - 1):
            self.setInitialState()
            [last_val, last_sol, last_evaluations] = self.runLocalSearchFast()#self.runLocalSearch(neigh_type)
            if last_val < best_val:
                best_val = last_val
                best_sol = last_sol
                best_evaluations = last_evaluations

        return best_val, best_sol, best_evaluations

    def runLocalSearch(self, neigh_type=neighType.Swap):
        best_val = self.costFunction(self.init_sol)              # Mejor valor
        best_sol = self.init_sol                                 # Mejor solucion
        improvement = True
        number_evaluations = 1
        while improvement:                                  # Mientras se mejore el valor de la funcion
            neighbors = neigh.Neighbors.getNeighbors(best_sol, neigh_type)              # Todos los vecinos
            n_neighbors = neighbors.shape[0]
            number_evaluations = number_evaluations + n_neighbors  # Se calcula es numero de evaluaciones
            best_val_among_neighbors = best_val
            for i in range(n_neighbors):                    # Se recorren todos los vecinos buscando el mejor
                sol = neighbors[i, :]
                fval = self.costFunction(sol)               # Se evalua la funcion
                if fval < best_val_among_neighbors:         # Si es mejor que el mejor valor entre los vecinos hasta el momento
                    best_val_among_neighbors = fval         # se actualiza el mejor valor
                    best_sol_among_neighbors = sol
            improvement = (best_val_among_neighbors < best_val) #  Se determina si ha habido mejora con respecto al ciclo anterior
            if improvement:
                best_val = best_val_among_neighbors           # Se actualiza el mejor valor y la mejor solucion
                best_sol = best_sol_among_neighbors
                #print(best_val, best_sol, number_evaluations)
        return best_val, best_sol, number_evaluations


    def runLocalSearchFast(self):
        # Parameters:
        maxFindIterations = self.model.numberOfElements
        maxEvaluations = 150
        continues = 3
        best_val = self.costFunction(self.init_sol)              # Mejor valor
        best_sol = self.init_sol                                 # Mejor solucion
        localMinima = False
        number_evaluations = 1
        n_neighbors = best_sol.shape[0]
        while not localMinima and number_evaluations < maxEvaluations:    # Mientras se mejore el valor de la funcion

            # init search permutation
            findIndexList = np.random.permutation(n_neighbors)  # Generamos una lista de indices a utilizar para la busqueda (se trata de aleatoreizar la busqueda)
            finding = 0
            betterFound = False
            targetIndex = np.random.randint(0, self.model.numberOfElements) # conseguimos el elemento a cambiar con el resto
            # quitamos el elemento de la lista
            delIndex = np.where(findIndexList == targetIndex)
            findIndexList = np.delete(findIndexList, delIndex)
            while not betterFound and finding < min(maxFindIterations, len(findIndexList)):
                new_best_sol = self.swap(best_sol, targetIndex, findIndexList[finding])
                new_best_val = self.costFunction(new_best_sol)
                number_evaluations += 1
                if new_best_val < best_val:
                    betterFound = True
                    best_sol = new_best_sol
                    best_val = new_best_val
                else:
                    finding += 1

            # Check if was improved
            if not betterFound:
                # Find 'k' random elements
                for i in range(continues):
                    continue_perm = np.random.permutation(self.model.numberOfElements)
                    continue_val = self.costFunction(continue_perm)
                    if continue_val < best_val:
                        localMinima = False
                        best_sol = continue_perm
                        best_val = continue_val
                        break
                    elif i == continues-1:
                        localMinima = True

            else:
                localMinima = False

        return best_val, best_sol, number_evaluations

    # Swap crea una vecindad basada en el operador de intercambio entre posiciones
    # Todas las permutaciones que se pueden obtener como un swap entre una posicion
    # y cualquiera de las restantes estan en la vecindad
    def swap(self, perm, aElement, bElement):
        n = perm.shape[0]
        neighbor = np.array(perm) # Guardaremos todos los vecinos en neighbors
        [neighbor[aElement], neighbor[bElement]] = [neighbor[bElement], neighbor[aElement]]
        return neighbor