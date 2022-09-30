"""
Created on Tue Sep 24 2022

@author: sasb

A python library for my physics lab course.
It can compute average, standard deviation, error spread, plot as fit functions.
"""

import numpy as np
import sympy as sp
from sympy.core.sympify import sympify
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def meas(array, alpha=0.025, outp=False, unit="", ddof=1):
    """Return the average and deviation of a measurement.
    Parameters:
        -array:
            array of the measurement results.
        -alpha:
            float, optional, default value alpha=0,025, significance percent of measurement.
        -outp:  
            boolean, optional, if outp=True, then it prints out the result in terminal.
        -unit:
            string, optional, if outp=True, then it also prints the unit.
    Returns:
        Tuple.
    Dependencies:
        Numpy, scipy.
    Raises:
        -"""
        
    mean = np.mean(array)
    N = np.shape(array)
    t0 = t.ppf(1-alpha, df=N)
    sigma = np.std(array, ddof=ddof)
    dev = (t0*sigma)/np.sqrt(N)
    if outp:
        print("(" + str("%.3f" % mean) + "+-" + str("%.3f" % dev) + ")" + unit)
    return mean, dev

def error_spread(model, params, arrays):
    """Calcualtes error spread from a model using sympy.
    Parameters:
        -model:
            string, model/formula that is used to calculate partial derivatives.
        -params:
            array of strings, array of parameters in the model equation.
        -arrays:
            array, array of measurement results.
    Return:
        float, spreaded error
    Dependencies:
        Sympy.
    Raises:
        -"""
    
    values = {}
    for i in range(len([params])):
        values[params[i]] = np.mean(arrays[i])
        
    stds = np.array([meas(arrays[i])[1] for i in range(len(params))])
    
    x = []
    expr = sympify(model)
    
    for i in range(len([params])):
        diff = sp.Derivative(expr, params[i]).doit()
        x.append(diff.evalf(subs=values)*stds[i])
        
    x = np.array(x, dtype=np.float64)
    
    return np.linalg.norm(x)

def plot(x, y, model=None, params=None, title=None, x_label=None, y_label=None, full_output=False):
    """Plots measurement results.
    Paraméterek:
        -x:
            array, x values
        -y:
            array, y values
        -model:
            string, optional, if we want to fit a function, we must give a model that is used for the fitting
        -params:
            array of strings, optional, if we want to fit a function, we must give an array of strings, that contain the parameters in the
            fitting model.
        -title:
            optional, title of plot
        -x_label:
            string, optional, label of x axis
        -y_label:
            string, optional, label of y axis
    Visszaad:
        -matplotlib.plot()
        -popt: optimális paraméterek az illeszéshez
        -perr: az illesztéshez használt paraméterek szórása (jelenleg nem működik)
    Dependencies:
        Numpy, matplotlib, scipy, sympy
    Current error:
        It can not compute the covariance matrix."""
        
    y_avg = np.array([np.mean(i) for i in y])
    error = np.array([meas(i)[1] for i in y])
    
    plt.errorbar(x, y_avg, yerr=error, c="r", fmt='o')
    plt.plot(x, y_avg)
    
    if model != None:
        temp = sp.lambdify(params, model, 'numpy')
        def f(*var): return temp(*var)
        p0 = np.array([1.0 for i in range(len(params)-1)])
        popt, pcov = curve_fit(f, x, y_avg, sigma=error, p0=p0, maxfev=4000)
        xplot = np.arange(np.min(x)-0.5, np.max(x)+1, 0.5)
        plt.plot(xplot, f(xplot, *popt))
        
    plt.grid()
    plt.show()
    
    perr = np.sqrt(np.diag(pcov))
    
    if full_output:
        print("fitted function parameters:")
        print([" " + params[i+1] + "=" + str("%.3f" % popt[i]) for i in range(len(params)-1)])
    
    return popt, perr, pcov
