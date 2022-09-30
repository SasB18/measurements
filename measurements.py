"""
Created on Tue Sep 24 2022

@author: sasb

Egy könyvtár, amely használható labor méréseinek kiértékelésére, hibák terjedésének kiszámolására, szimbolikus 
egyenletek megadására, függvények illesztésére, stb.
"""

import numpy as np
import sympy as sp
from sympy.core.sympify import sympify
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def meas(array, alpha=0.025, outp=False, unit="", ddof=1):
    """Megadja egy mérési eredmény középértékét és eltérését.
    Paraméterek:
        -array: lista amelyben a mérés eredményei szerepelnek.
        -alpha: float, opcionális, megadja a szignifikancia szintet. Alap értéke alpha=0,025.
        -outp: boolean, opcionális, ha outp=True akkor kiírja az értéket a terminálba.
        -unit: string, opcionális, outp=True esetén mértékegységet is ír.
    Visszaad:
        Tuple.
    Dependencies:
        Numpy, scipy."""
        
    mean = np.mean(array)
    N = np.shape(array)
    t0 = t.ppf(1-alpha, df=N)
    sigma = np.std(array, ddof=ddof)
    dev = (t0*sigma)/np.sqrt(N)
    if outp:
        print("(" + str("%.3f" % mean) + "+-" + str("%.3f" % dev) + ")" + unit)
    return mean, dev

def error_spread(model, params, arrays):
    """Kiszámolja a hibaterjedését egy függő mennyiségnek egy modell alapján. A sympy könyvtár szimbolikus deriválását
    használva deriválja a modellt, majd kiértékeli a mért mennyiségek középértékénél.
    Paraméterek:
        -model: Modell/képlet amely alapján számolja a hibaterjedést. Úgy kell megadni, hogy a sympy
        kifejezhető összefüggéssé tudja alakítani.
        -params: Az egyenletben szereplő paraméterek listája stringként. Ezek azok a paraméterek amelyek szerint le fogja
        deriválni a modellt.
        -arrays: Az adathalmazok listája. Ebből meghatározza a középértékeket és az eltéréseket is.
    Visszaad:
        float, a függő mennyiség hibája
    Dependencies:
        Sympy."""
    
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
    """Egy mérési eredményt ábrázol grafikonon hibasávval.
    Paraméterek:
        -x: az x tengely értékei
        -y: az y tengely értékei (mért eredmények)
        -model: opcionális, ha függvényillesztést szeretne csinálni, akkor meg kell adni egy modellt, amely alapján
        végezné az illesztést. (Jelenleg csak egyenest tud)
        -params: opcionális, ha függvényilleszészt szeretne csinálni, akkor ebbe a paraméterbe kell listaként megadni
        a modellben szereplő változókat. Fontos, hogy 
        -title: opcionális, címet ad a grafikonnak
        -x_label: opcionális, címkét as az x tengelynek
        -y_label: opcionális, címkét ad az y tengelynek
    Visszaad:
        -matplotlib.plot()
        -popt: optimális paraméterek az illeszéshez
        -perr: az illesztéshez használt paraméterek szórása (jelenleg nem működik)
    Dependencies:
        Numpy, matplotlib, scipy, sympy
    Probléma:
        A sympy.lambdify miatt nem tudja kiszámolni a kovarianciamátrixot."""
        
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
        print("Az illesztett függvény paraméterei:")
        print([" " + params[i+1] + "=" + str("%.3f" % popt[i]) for i in range(len(params)-1)])
    
    return popt, perr, pcov


if __name__ == "__main__":
    """ Példák a fent definiált függvényekre """
    
    # HIBATERJEDÉS
    model = "T**2+y**3"
    z = error_spread(model, ["T", "y"], [[2, 2.3, 2.2], [1, 1.2, 0.9]])    
    
    # FÜGGVÉNYILLESZTÉS
    # x = np.array([1, 2, 3, 4])
    # y = np.array([[1.1, 1.2, 0.8], [3.87, 4.03, 4.4], [9, 9.1, 8.7], [16.2, 16.3, 16.8]])
    # f = "exp(a*x-b)*c+d" # Innen folytatni (modell megadása és illesztés)
    # popt, perr, pcov = plot(x, y, model=f, params=["x", "a", "b", "c", "d"], full_output=True)

    pass