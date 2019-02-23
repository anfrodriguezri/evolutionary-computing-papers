"""
	Class for objective functions
"""

class ObjectiveFunction():
	def __init__(self, type, limits, f=lambda x: x):
		self.type = type
		self.f = f
		self.limits = limits
