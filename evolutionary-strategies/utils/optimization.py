import math
import operator

import numpy as np


class Optimization():
    MAX = 'maximize'
    MIN = 'minimize'

    @staticmethod
    def get_operator(objective, comparison, equality=False):
        if objective == Optimization.MAX and comparison == 'worse' \
                or objective == Optimization.MIN and comparison == 'better':
            if not equality:
                return operator.lt
            else:
                return operator.le
        elif objective == Optimization.MAX and comparison == 'better' \
                or objective == Optimization.MIN and comparison == 'worse':
            if not equality:
                return operator.gt
            else:
                return operator.ge

    @staticmethod
    def is_worse(value_a, value_b, objective):
        op = Optimization.get_operator(objective, 'worse')
        return op(value_a, value_b)

    @staticmethod
    def is_worse_or_equal(value_a, value_b, objective):
        op = Optimization.get_operator(objective, 'worse', equality=True)
        return op(value_a, value_b)

    @staticmethod
    def is_better(value_a, value_b, objective):
        op = Optimization.get_operator(objective, 'better')
        return op(value_a, value_b)

    @staticmethod
    def is_better_or_equal(value_a, value_b, objective):
        op = Optimization.get_operator(objective, 'better', equality=True)
        return op(value_a, value_b)

    @staticmethod
    def get_worse(population, n, objective):
        population.sort()

        if objective == Optimization.MAX:
            return population[:n]
        elif objective == Optimization.MIN:
            return population[-n:]
