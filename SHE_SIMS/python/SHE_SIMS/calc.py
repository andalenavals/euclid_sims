"""
A module to hold some maths or statistics oriented functions woring on numpy arrays.
"""

import numpy as np

def rotg(g1, g2, deg):
	"""
	Returns the rotated (g1, g2) by deg degrees.
	Be careful, there is this "extra" factor 2 here, given the meaning of g1 and g2.
	"""
	
	complexg = g1 + g2*1j
	rotg = complexg * np.exp(1j * 2.0 * np.pi*deg/180.0)
	return (rotg.real, rotg.imag)
