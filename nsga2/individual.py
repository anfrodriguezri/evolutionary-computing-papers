"""
	Class for individuals
"""

class Individual():
	def __init__(self, chromosome):
		self.chromosome = chromosome
		self.rank = None
		self.distance = 0

	def eval(self, f):
		return f(self.chromosome)

