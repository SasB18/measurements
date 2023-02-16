import numpy as np
from numpy.polynomial import Polynomial

linear = lambda x, m, b: m*x + b

def polynomial(x, *deg):
    p = Polynomial([*deg])
    return p(x)

def sinusiod(x, A, f, a):
    return A*np.sin(x*f+a)

def exponential(x, A, b):
    return A*np.exp(x*b)

def gauss(x, m, s):
    return (1/(s*np.sqrt(2*np.pi))) * np.exp((-1/2) * ( (x-m) /s)**2)

def cauchy(x, x0, g):
    return (1/np.pi) * (g**2 / ((x-x0)**2 + g**2))

def laplace(x, m, b):
    return (1/(2*b)) * np.exp(-(np.abs(x-m))/b)