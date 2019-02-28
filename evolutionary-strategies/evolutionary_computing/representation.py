import numpy as np


from bitarray import bitarray
from utils.optimization import Optimization
from functools import cmp_to_key


class Individual():
    def __init__(self, genome_length=50, genome=None):
        if not genome:
            self.genome = np.random.randint(
                2, size=genome_length).tolist()
            self.genome = bitarray(self.genome)
        else:
            self.genome = genome

        self.genome_length = self.genome.length()
        self.fitness = self.genome.count()

    def __repr__(self):
        return self.genome.to01()


class ESIndividual(Individual):
    def __init__(self, Y=None, S=None, optimization_function=None):
        self.Y = Y  # strategy parameters (dimensions)
        self.S = S  # object parameters (sigmas)

        if Y is None and optimization_function:
            self.Y = np.random.uniform(optimization_function.domain['lower'],
                                       optimization_function.domain['upper'],
                                       optimization_function.dim)

        if S is None and optimization_function:
            self.S = np.random.rand(optimization_function.dim)

        self.fitness = optimization_function.eval(self.Y) \
            if optimization_function else None

    def __repr__(self):
        return "Y" + str(self.Y) + ":S" \
            + str(self.S) + ":fitness " + str(self.fitness)


class Population():
    def __init__(self, size=0,
                 individuals=None,
                 individual_factory=Individual,
                 **kwargs):
        self.individuals = individuals or []
        length = len(self.individuals)

        if length != size:
            for i in range(size):
                self.individuals.append(individual_factory(**kwargs))

        self.size = len(self.individuals)
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.individuals[key]

    def __setitem__(self, key, value):
        if type(value) is not Individual:
            raise TypeError('Only Individual class can be added to Population')

        self.individuals[key] = value

    def __delitem__(self, key):
        del self.individuals[key]
        self.size -= 1

    def __len__(self):
        return self.size

    def __str__(self):
        string = ""
        for individual in self.individuals:
            string += str(individual) + '\n'

        return string

    def __setattr__(self, name, value):
        if name == 'individuals':
            self.size = len(value)

        super().__setattr__(name, value)

    def append(self, individual):
        self.individuals.append(individual)
        self.size += 1

    def extend(self, population):
        for individual in population:
            self.append(individual)
            self.size += 1

    def remove(self, individual):
        self.individuals.remove(individual)
        self.size -= 1

    def sort(self, reverse=False):
        def individual_cmp(individual_a, individual_b):
            return individual_b.fitness - individual_a.fitness

        self.individuals.sort(key=cmp_to_key(individual_cmp), reverse=reverse)
