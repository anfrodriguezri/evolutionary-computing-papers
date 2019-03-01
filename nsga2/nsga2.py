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
def pop_init(pop_size, num_dims, limits):
	min_lim, max_lim = limits
	return [Individual(np.random.uniform(min_lim, max_lim, size=num_dims)) for i in range(pop_size)]

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
	cross_point = np.random.randint(1, len(parent1.chromosome))

	child1 = Individual(
		chromosome = np.append(parent1.chromosome[:cross_point], parent2.chromosome[cross_point:]),
	)
	child2 = Individual(
		chromosome = np.append(parent2.chromosome[:cross_point], parent1.chromosome[cross_point:]),
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
	for p in P:
		gen_q = (x for x in P if x != p)
		n[p] = 0
		S[p] = []
		for q in gen_q:
			if dominate(objectives, p, q):
				S[p].append(q)
			elif dominate(objectives, q, p):
				n[p] += 1
		if n[p] == 0:
			F[1].append(p)
			p.rank = 1

	i = 1
	while len(F[i]) != 0:
		H = []
		for p in F[i]:
			for q in S[p]:
				n[q] -= 1
				if n[q] == 0: 
					H.append(q)
					q.rank = i + 1
		i += 1
		F[i] = H
	return F

def crowding_distance_assignment(I, objectives):
	l = len(I)
	for m in objectives:
		I.sort(key=lambda x: m.f(x.chromosome))
		I[0].distance = I[-1].distance = float('inf')
		for i in range(1, l-1):
			I[i].distance += m.f(I[i+1].chromosome) - m.f(I[i-1].chromosome)

def main_loop(T, N, num_dims, objectives, limits):
	P = pop_init(N, num_dims, limits)
	F = fast_non_dominated_sort(P, objectives)

	selected = tournament(P, num_parents=20, k=2)
	Q = []
	for i in range(N // 2):
		parent1, parent2 = np.random.choice(selected, size=2)

		child1, child2 = recombination(parent1, parent2)
		child1 = mutation(child1, 0.1)
		child2 = mutation(child2, 0.1)

		Q += [child1, child2]

	for t in range(1, T):
		R = P + Q
		F = fast_non_dominated_sort(R, objectives)
		P_next = []
		Q_next = []

		i = 1
		while len(P_next) < N:
			crowding_distance_assignment(F[i], objectives)
			P_next += F[i]
			i += 1

		P_next.sort(key=lambda x: (x.rank, x.distance))
		P_next = P_next[:N]
		for i in range(N // 2):
			parent1, parent2 = np.random.choice(selected, size=2)

			child1, child2 = recombination(parent1, parent2)
			child1 = mutation(child1, 0.1)
			child2 = mutation(child2, 0.1)

			Q_next += [child1, child2]

		P = P_next
		Q = Q_next

		print([p.rank for p in P])

	return P




	
