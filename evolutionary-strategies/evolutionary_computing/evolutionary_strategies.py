import math
import copy
import functools

import numpy as np

from evolutionary_computing.representation import *
from evolutionary_computing.operators import *


class EvolutionaryStrategy():
    def __init__(self,
                 optimization_function,
                 population_size,
                 total_parents,
                 selection_type,
                 total_offspring,
                 individual_factory=ESIndividual,
                 learning_rate=None,
                 objective='minimize',
                 max_iterations=100):
        self.generation = 0
        self.best_solution = None
        self.optimization_function = optimization_function
        self.individual_factory = individual_factory
        self.selection_type = selection_type
        self.population_size = population_size
        self.total_parents = total_parents

        if selection_type == ',' and total_offspring <= population_size:
            raise Exception('Comma replacement requires total_offspring ' +
                            'greater than population_size')

        self.total_offspring = total_offspring

        self.learning_rate = learning_rate or 1 / \
            math.sqrt(2 * population_size)
        self.objective = objective

        self.max_iterations = max_iterations

    def initialize_population(self):
        population = Population(
            size=self.population_size,
            individual_factory=self.individual_factory,
            optimization_function=self.optimization_function)

        return population

    def fitness_function(self, individual):
        individual.fitness = self.optimization_function.eval(
            individual.Y)

        return individual.fitness

    def evaluate(self, population):
        population.fitness = 0

        if not self.best_solution:
            if self.objective == Optimization.MAX:
                best_fitness = -math.inf
            elif self.objective == Optimization.MIN:
                best_fitness = math.inf
        else:
            best_fitness = self.best_solution.fitness

        for idx, individual in enumerate(population):
            indv_fitness = self.fitness_function(individual)
            population.fitness += indv_fitness

            if Optimization.is_better(indv_fitness,
                                      best_fitness,
                                      self.objective):
                best_fitness = indv_fitness
                self.best_solution = copy.deepcopy(individual)

        return population.fitness

    def empty_population(self):
        return Population(individual_factory=self.individual_factory)

    def termination_reached(self):
        # if self.best_solution is not None \
        #         and self.optimization_function \
        #                 .close_to_optimal(self.best_solution.Y):
        #     print('optimal found')
        #     return True

        if self.generation >= self.max_iterations:
            return True
        return False

    def s_recombination(self, marriage):
        p = len(marriage)
        return functools.reduce(lambda x, y: x.S + y.S, marriage) / p

    def y_recombination(self, marriage):
        p = len(marriage)
        return functools.reduce(lambda x, y: x.Y + y.Y, marriage) / p

    def s_mutation(self, S):
        return S * np.exp(self.learning_rate * np.random.randn(S.shape[0]))

    def y_mutation(self, Y, S_prime):
        return Y + S_prime * np.random.randn(S_prime.shape[0])

    def run(self):
        fitness = np.zeros(self.max_iterations)
        current_population = self.initialize_population()

        self.generation = 0
        while not self.termination_reached():
            # print(self.generation)
            # print(current_population)
            self.evaluate(current_population)
            fitness[self.generation] = self.best_solution.fitness

            offspring_population = self.empty_population()

            for l in range(self.total_offspring):
                marriage = RandomSelection.operate(
                    current_population, self.total_parents)
                Sl = self.s_recombination(marriage)
                Yl = self.y_recombination(marriage)

                Sl = self.s_mutation(Sl)
                Yl = self.y_mutation(Yl, Sl)

                new_individual = self.individual_factory(Yl, Sl)
                offspring_population.append(new_individual)

            self.evaluate(offspring_population)

            if self.selection_type == ',':
                new_population = CommaSelection.operate(
                    offspring_population,
                    self.population_size,
                    self.objective
                )
            elif self.selection_type == '+':
                new_population = PlusSelection.operate(
                    current_population,
                    offspring_population,
                    self.population_size,
                    self.objective
                )

            current_population = new_population

            self.generation += 1

        return fitness
