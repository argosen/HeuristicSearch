__author__ = 'goe'

import dataModel as dm
import numpy as np
import Neighbors as neigh
from Neighbors import NeighborType as neighType
from enum import Enum

import random
from deap import algorithms, base, creator, tools

class MateType(Enum):
    HalfPreserve = 1
    HalfPreserve2 = 2
    Interchange1 = 3


class GeneticAlgorithm:

    def __init__(self, filename, mateFunction=3):
        self.model = dm.DataModel(filename)
        self.result = np.array(range(self.model.numberOfElements))
        self.init_sol = np.random.permutation(self.model.numberOfElements)
        self.population = 1
        self.cxpb = 0
        self.mutpb = 0
        self.ngen = 0

        self.valueList = []

        self.setConfiguration()

        # CREATE THE MAIN OBJECTS
        # Se crea una clase FitnessMax para la maximizacion de funciones
        creator.create("LocationsMin", base.Fitness, weights=(-1,)) # Esta funcion genera una clase de tipo 'LocationsMin' que hereda de 'base.Fitness'. El ultimo parametro define el buscar el minimo en las iteraciones.
        # Se crea una clase individuo asociada a la clase LocationsMin
        creator.create("Individual", list, fitness=creator.LocationsMin) # crea la definicion de individuo utilizando la clase creada justo antes.
        # Heredamos las clases y funciones implementadas como parte de DEAP

        # CREATE THE TOOLBOX OBJECT
        self.toolbox = base.Toolbox()
        # Utilizaremos una representacion binaria
        self.toolbox.register("numberPermutation", self.pickNewValue) # attaches the alias 'numberPermutation' to the function parsed 'np.random.permutation'. The rest of parameters are values to pass to teh function
        # Definimos que nuestros individuos tendran 'self.model.numberOfElements' variables enteras
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.numberPermutation, n=self.model.numberOfElements) # Repit 'n' times the function defined in the toolbox.
        # Definimos la poblacion a partir de los individuos
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) # Similar to previous, but the missing n value is given later in the execution.
        # Asociamos como funcion de aptitud la funcion OneMax
        self.toolbox.register("evaluate", self.costFunction)
        # El operador de mutacion cambiara 1-->0  y 0-->1 con una probabilidad de mutacion de 0.05
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) # !!!
        # Usaremos seleccion por torneo con un parametro de torneo = 3
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # SELECT THE MATE FUNCTION
        if mateFunction == 1:
            # self.toolbox.register("mate", tools.cxTwoPoint) # !!!
            self.toolbox.register("mate", self.cxHalfAndOrderRest) # !!!
        elif mateFunction == 2:
            self.toolbox.register("mate", self.cxHalfAndOrderRest2) # !!!
        elif mateFunction == 3:
            self.toolbox.register("mate", self.cxRandomItemExChange) # !!!

    def pickNewValue(self):
        if len(self.valueList) == 0:
            self.valueList = np.random.permutation(self.model.numberOfElements)

        value = self.valueList[0]
        self.valueList = self.valueList[1:]

        return value

    def cxHalfAndOrderRest(self, ind1, ind2):
        """Executes a custom crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.

        The first half of the individual is mantained, and the second half is
        generated using the rest of the unused numbers in the same order as in
        individual 2. Same process is applied to the second individual.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        size = min(len(ind1), len(ind2))
        half_size = size / 2

        ind1_aux = ind1[:]
        ind2_aux = ind2[:]

        # erase the elements in ind1 from the aux list of ind2
        for element in ind1[:half_size]:
            ind2_aux.remove(element)
        # erase the elements in ind2 from the aux list of ind1
        for element in ind2[:half_size]:
            ind1_aux.remove(element)

        for i in range(len(ind2_aux)):
            ind1[i+half_size] = ind2_aux[i]

        for i in range(len(ind1_aux)):
            ind2[i+half_size] = ind1_aux[i]

        return ind1, ind2

    def cxHalfAndOrderRest2(self, ind1, ind2):
        """Executes a custom crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.

        The first half of the individual is mantained, and the second half is
        generated using the rest of the unused numbers in the same order as in
        individual 2. Same process is applied to the second individual.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        size = min(len(ind1), len(ind2))
        half_size = size / 2

        ind1_aux = ind1[:]
        ind2_aux = ind2[:]

        # erase the elements in ind1 from the aux list of ind2
        for element in ind1[:half_size]:
            ind2_aux.remove(element)
        # erase the elements in ind2 from the aux list of ind1
        for element in ind2[half_size:]:
            ind1_aux.remove(element)

        for i in range(len(ind2_aux)):
            ind1[i+half_size] = ind2_aux[i]

        for i in range(len(ind1_aux)):
            ind2[i] = ind1_aux[i]

        return ind1, ind2

    def cxRandomItemExChange(self, ind1, ind2):
        """Executes a custom crossover on the input :term:`sequence`
        individuals. The two individuals are modified in place and both keep
        their original length.

        Select an item randomly, and

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.randint` function from the Python
        base :mod:`random` module.
        """
        size = min(len(ind1), len(ind2))

        original1 = ind1[:]
        original2 = ind2[:]

        position = np.random.random_integers(0, size-1)
        # get the element to place in the position
        elementInPosition1 = ind1[position]
        elementInPosition2 = ind2[position]
        # Find if the new element is in other position
        elementIndexToMoveIn1 = ind1.index(elementInPosition2)
        elementIndexToMoveIn2 = ind2.index(elementInPosition1)

        if elementIndexToMoveIn1 != position:
            ind1[elementIndexToMoveIn1] = elementInPosition1
            ind1[position] = elementInPosition2

        if elementIndexToMoveIn2 != position:
            ind2[elementIndexToMoveIn2] = elementInPosition2
            ind2[position] = elementInPosition1

        return ind1, ind2

    def setConfiguration(self, population=150, cxpb=0.5, mutpb=0.2, ngen=12):
        self.population = population
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen

    def costFunction(self, perm):
        res = 0
        n = self.model.numberOfElements

        for i in range(n):
            for j in range(n):
                sig_i = perm[i]
                sig_j = perm[j]
                res = res + self.model.distanceMatrix[sig_i, sig_j] * self.model.flowMatrix[i, j]

        return res, # The comma is needed because the result of the cost function must be a sequence of elements with DEAP

    def solve(self, n_starts=100):
        [best_val, best_sol, best_evaluations] = self.runGeneticAlgorithm()
        for i in range(n_starts - 1):
            [last_val, last_sol, last_evaluations] = self.runGeneticAlgorithm()
            if last_val < best_val:
                best_val = last_val
                best_sol = last_sol
                best_evaluations = last_evaluations

        return best_val, best_sol, best_evaluations

    def runGeneticAlgorithm(self):
        # La poblacion tendra 150 individuos
        pop = self.toolbox.population(n=self.population)

        verbose = False
        # El algoritmo evolutivo simple utiliza los siguientes parametros
        # Probabilidad de cruzamiento 0.5
        # Probabilidad de aplicar el operador de mutacion 0.2
        # Numero de generaciones 10
        if verbose:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            algorithms.eaSimple(pop, self.toolbox, stats=stats, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, verbose=verbose)
        else:
            algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, verbose=verbose)

        best_sol = tools.selBest(pop, k=1)
        best_val = self.costFunction(best_sol[0])
        number_evaluations = self.ngen

        print(best_val, best_sol, number_evaluations)

        return best_val, best_sol, number_evaluations