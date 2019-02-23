import sys
import numpy as np
from individual import Individual
from objective_function import ObjectiveFunction

# Selection Method
def tournament(pop, num_parents, k=20):
    n = len(pop)
    selected_parents = []
    for i in range(num_parents):
        k_pop = np.random.choice(pop, size=k)
        best = min(k_pop, key=lambda x: x.rank)
        selected_parents.append(best)
    return selected_parents

# Initialize population
def pop_init(pop_size, limits):
	min_lim, max_lim = limits
	return [Individual(np.random.uniform(min_lim, max_lim)) for i in range(pop_size)]

"""
	Genetic Operators
"""

# Mutation
def mutation(individual, sigma):
	offspring = np.array(individual.chromosome[:])
	pos = np.random.randint(0, len(individual.chromosome))
	offspring[pos] += np.random.randn() * sigma 
	return Individual(offspring)

# Recombination
def recombination(parent1, parent2):
	child1 = Individual(
		chromosome = np.append(parent1[:crossPoint], parent2[crossPoint:]),
	)
	child2 = Individual(
		chromosome = np.append(parent2[:crossPoint], parent1[crossPoint:]),
	)
	
	return [ child1, child2 ]


"""
	NSGA2 Methods
"""

def dominate(objectives, x, y):
	count = 0
	for objective in objectives:
		f_x = x.eval(objective.f)
		f_y = y.eval(objective.f)
		if objective.type == 'max':
			if f_x < f_y:
				return False
			elif f_x == f_y:
				continue
			else:
				count += 1
		else:
			if f_x > f_y:
				return False
			elif f_x == f_y:
				continue
			else:
				count += 1
	return count >= 1


def fast_non_dominated_sort(P, objectives):
	F, n, S = {}, {}, {}
	F[1] = []
	for p in range(len(P)):
		gen_q = (x for x in range(len(P)) if x != p)
		n[p] = 0
		S[p] = []
		for q in gen_q:
			if dominate(objectives, P[p], P[q]):
				S[p].append(q)
			elif dominate(objectives, P[q], P[p]):
				n[p] += 1
		if n[p] == 0:
			F[1].append(p)
			P[p].rank = 1

	i = 1
	while len(F[i]) != 0:
		H = []
		for p in F[i]:
			for q in S[p]:
				n[q] -= 1
				if n[q] == 0: 
					H.append(q)
					P[q].rank = i + 1
		i += 1
		F[i] = H
	return F

def crowding_distance_assignment(I, objectives):
	l = len(I)
	I = [(i, 0) for i in I]
	for m in objectives:
		I.sort(key=lambda x: m(x[0]))
		I[0] = I[-1] = float('inf')
		for i in range(1, l-2):
			I[i][1] += m(I[i+1][0]) - m(I[i-1][0])

def main_loop(pop_size, num_dims, objectives, limits):
	P = pop_init(pop_size, limits)
	F = fast_non_dominated_sort(P, objectives)

	
	selected = tournament(P, num_parents=20, k=2)
	
