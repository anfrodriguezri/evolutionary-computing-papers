import math
import numpy as np
from numbers import Number


class Function:
    def __init__(self, dim=1):
        self.dim = dim
        self.domain = {'lower': -math.inf, 'upper': math.inf}

    def eval(self, X):
        pass

    def feasible(self, X):
        if isinstance(X, Number):
            X = [X]

        for i in range(self.dim):
            if X[i] > self.domain['upper'] or X[i] < self.domain['lower']:
                return False
        return True


class DiffFunction(Function):
    def gradient(self, X):
        pass


class Diff2Function(DiffFunction):
    def hessian(self, X):
        pass

    def hessian_det(self, X):
        return np.linalg.det(self.hessian(X))


class QuadraticFunction(Diff2Function):
    def eval(self, X):
        e = 0

        for i in range(self.dim):
            e += (X[i]**2)

        return e

    def gradient(self, X):
        grad = np.zeros(self.dim)

        for i in range(self.dim):
            grad[i] = 2 * X[i]

        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])

        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    hess[i][j] = 2

        return hess


class RastriginFunction(Diff2Function):
    def __init__(self, dim=1):
        super(RastriginFunction, self).__init__(dim)
        self.domain = {'lower': -5.12, 'upper': 5.12}
        self.name = 'Rastrigin Function'

    def eval(self, X):
        e = 10 * self.dim

        for i in range(self.dim):
            e += (X[i]**2) - 10 * math.cos(2 * math.pi * X[i])

        return e

    def gradient(self, X):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            grad[i] = 2 * (X[i] + 10 * math.pi * math.sin(2 * math.pi * X[i]))

        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])

        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    hess[i][j] = 2 * (1 + 20 * (math.pi**2) *
                                      math.cos(2 * math.pi * X[i]))

        return hess

    def close_to_optimal(self, X):
        for i in range(self.dim):
            if abs(X[i] - 0) > 0.01:
                return False

        return True


class GriewankFunction(Diff2Function):
    def __init__(self, dim=1):
        super(GriewankFunction, self).__init__(dim)
        self.domain = {'lower': -600, 'upper': 600}
        self.name = 'Griewank Function'

    def eval(self, X):
        suma = 0
        prod = 1

        for i in range(self.dim):
            suma += (X[i]**2) / 4000
            prod *= math.cos(X[i] / math.sqrt(i + 1))

        return suma - prod + 1

    def gradient(self, X):
        grad = np.zeros(self.dim)

        for i in range(self.dim):
            prod = 1
            for k in range(self.dim):
                if k != i:
                    prod *= math.cos(X[i] / math.sqrt(i + 1))

            grad[i] = (X[i] / 2000) + math.sin(X[i] /
                                               math.sqrt(i + 1)) * (1 / math.sqrt(i + 1)) * prod

        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])

        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    prod = 1

                    for k in range(self.dim):
                        if k != i:
                            prod *= math.cos(X[k] / math.sqrt(k + 1))

                    hess[i][j] = (1 / 2000) + math.cos(X[i]) * \
                        (1 / (i + 1)) * prod
                else:
                    prod = 1

                    for k in range(self.dim):
                        if k != i and k != j:
                            prod *= math.cos(X[k] / math.sqrt(k + 1))

                    hess[i][j] = math.sin(
                        X[i] / math.sqrt(i + 1)) * (1 / math.sqrt(i + 1)) * prod
                    hess[i][j] *= -1 * \
                        math.sin(X[j] / math.sqrt(j + 1)) * \
                        (1 / math.sqrt(j + 1))

        return hess

    def close_to_optimal(self, X):
        for i in range(self.dim):
            if abs(X[i] - 0) > 0.01:
                return False

        return True


class RosenbrockFunction(Diff2Function):
    def __init__(self, dim=1):
        super(RosenbrockFunction, self).__init__(dim)
        self.domain = {'lower': -2048, 'upper': 2048}
        self.name = 'Rosenbrock Function'

    def eval(self, X):
        suma = 0

        for i in range(self.dim - 1):
            suma += (100 * (X[i + 1] - X[i]**2)**2) + (1 - X[i])**2

        return suma

    def gradient(self, X):
        grad = np.zeros(self.dim)

        for i in range(self.dim):
            if(i != self.dim - 1):
                grad[i] += 200 * (X[i + 1] - (X[i]**2)) * (-2 * X[i])

            if(i != 0):
                grad[i] += 200 * (X[i] - (X[i - 1]**2))

            grad[i] += 2 * (X[i] - 1)

        return grad

    def hessian(self, X):
        hess = np.zeros([self.dim, self.dim])

        for i in range(self.dim):
            for j in range(self.dim):

                if (i < j - 1) or (i > j + 1):
                    continue

                if (i == j - 1):
                    if i != 0:
                        hess[i][j] += -400 * X[j - 1]
                elif i == j:
                    if (i != self.dim - 1):
                        hess[i][j] += -400 * (X[j + 1] - (3 * X[j]**2))

                    if i != 0:
                        hess[i][j] += 200

                    hess[i][j] += 2
                elif (i == j + 1):
                    if (i != self.dim - 1):
                        hess[i][j] += -400 * X[j]

        return hess


class SchwefelFunction(Function):
    def __init__(self, dim=1):
        super(SchwefelFunction, self).__init__(dim)
        self.domain = {'lower': -500, 'upper': 500}
        self.name = 'Schwefel Function'

    def eval(self, X):
        suma = 0

        for i in range(self.dim):
            suma += X[i] * math.sin(math.sqrt(abs(X[i])))

        return 418.9829 * self.dim - suma


class OneMaxFunction(Function):
    def eval(self, x):
        return x.count()

    def feasible(self, x):
        return True


class M1(Function):
    def __init__(self):
        self.dim = 1
        self.domain = {'lower': 0, 'upper': 1}

    def eval(self, x):
        return math.sin(5 * math.pi * x) ** 6


class M2(Function):
    def __init__(self):
        self.dim = 1
        self.domain = {'lower': 0, 'upper': 1}

    def eval(self, x):
        e_exponent = -2
        e_exponent *= math.log(2)
        e_exponent *= ((x - .1) / .8)**2

        res = math.e**e_exponent
        res *= math.sin(5 * math.pi * x) ** 6

        return res


class M3(Function):
    def __init__(self):
        self.dim = 1
        self.domain = {'lower': 0, 'upper': 1}

    def eval(self, x):
        return math.sin(5 * math.pi * (x**.75 - .05)) ** 6


class M4(Function):
    def __init__(self):
        self.dim = 1
        self.domain = {'lower': 0, 'upper': 1}

    def eval(self, x):
        e_exponent = -2
        e_exponent *= math.log(2)
        e_exponent *= ((x - .08) / .854)**2

        res = math.e**e_exponent
        res *= math.sin(5 * math.pi * (x**.75 - .05)) ** 6

        return res


class PrisonersDilema(Function):
    def __init__(self):
        self.dim = 2
        self.domain = {'lower': 0, 'upper': 1}

    def eval(self, A, B):
        if not A and not B:
            return [4, 4]
        elif A and not B:
            return [7, 1]
        elif not A and B:
            return [1, 7]
        else:
            return [2, 2]
