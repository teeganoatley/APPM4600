import numpy as np
import math
from numpy.linalg import norm 
from numpy.linalg import inv

def driver():

    x0 = np.array([1, 1])
    arr = np.array([[1/16, 1/18], [0, 1/6]])
    
    tol = 1e-10
    Nmax = 1000
    
    [xstar,ier,its] = iter(x0,arr,tol,Nmax)
    
    print(xstar)
    print('Part a: Error message reads:',ier)
    print('Part a: Number of iterations is:',its)
    
    [xstar,ier,its] = Newton(x0,tol,Nmax)
    
    print(xstar)
    print('Part c: Error message reads:',ier)
    print('Part c: Number of iterations is:',its)

def evalF(x): 

    F = np.zeros(2)
    g1 = 3*(x[0]*(x[1]**2))
    g2 = x[0]**3
    
    F[0] = 3*x[1]**2 - x[1]**2
    F[1] = g1 - g2 - 1

    return F
    
def evalJ(x): 
    
    J = np.array([[6*x[0]**2, -2*x[1]],[3*x[1]**2-3*x[0]**2, 6*x[0]*x[1]]])
    
    return J
    
def iter(x0, arr, tol, Nmax):
   for its in range(Nmax):
       F = evalF(x0)
       x1 = x0 - arr.dot(F)
       
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
   
def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]
    
driver()