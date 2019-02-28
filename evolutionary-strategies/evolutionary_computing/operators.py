import numpy as np

from utils.optimization import Optimization


class RandomSelection():
    def operate(population, p):
        if p == population.size:
            return population

        parents = []

        for i in range(p):
            rand = np.random.randint(population.size)

            parent = population[rand]
            parents.append(parent)

        return parents


class CommaSelection():
    @staticmethod
    def operate(candidate_population, n, objective):
        candidate_population.sort(n)

        old_individuals = candidate_population.individuals

        if objective == Optimization.MAX:
            candidate_population.individuals = old_individuals[-n:]
        elif objective == Optimization.MIN:
            candidate_population.individuals = old_individuals[:n]

        return candidate_population


class PlusSelection():
    @staticmethod
    def operate(current_population, candidate_population, n, objective):
        current_population.extend(candidate_population)
        current_population.sort(n)

        old_individuals = current_population.individuals

        if objective == Optimization.MAX:
            current_population.individuals = old_individuals[-n:]
        elif objective == Optimization.MIN:
            current_population.individuals = old_individuals[:n]

        return current_population
