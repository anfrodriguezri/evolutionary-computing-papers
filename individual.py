"""
	Class for individuals
"""

class Individual():
	def __init__(self, chromosome):
		self.chromosome = chromosome
		self.rank = None
		self.distance = None

	def eval(self, f):
		return f(self.chromosome)

