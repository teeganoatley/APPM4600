import numpy as np
import math
from numpy.linalg import norm 
from numpy.linalg import inv

def driver():

    x0 = np.array([1, 1, 1])
    
    tol = 1e-10
    Nmax = 1000
    
    [xstar,ier,its] = iter(x0,tol,Nmax)
    
    print(xstar)
    print('Part a: Error message reads:',ier)
    print('Part a: Number of iterations is:',its)
    

def evalF(x): 

    F = x[0]**2 + 4*x[1]**2 + 4*x[2]**2 - 16
    
    Fpart = np.zeros(3)
    Fpart[0] = 2*x[0]
    Fpart[1] = 8*x[1]
    Fpart[2] = 8*x[2]

    return F, Fpart
    
def iter(x0, tol, Nmax):
   for its in range(Nmax):
       F,Fpart = evalF(x0)
       x1 = x0 - (F/(Fpart[0]**2+Fpart[1]**2+Fpart[2]**2))*Fpart
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier, its]
       elif (norm(x1-x0) > 1/tol):
           print("exploded")
           xstar = [np.nan, np.nan]
           ier = 1
           return[xstar, ier, its]
           
       x0 = x1
    
   xstar = x1
   ier = 1
    
   return[xstar,ier,its]
    
driver()