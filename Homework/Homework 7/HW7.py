# Homework 6
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver(): 
    # Problem 1
    N = 100
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    
    h = 2/(N-1)

    x = np.zeros(N)
    y = np.zeros(N)

    for i in range(0,N): 
        x[i] = -1 + (i-1)*h
        y[i] = f(x[i])
        
    [c, p] = coeff(x,y,N,a,b)
    
    plt.figure()
    plt.plot(x,y,'o')
    plt.plot(x,p)
    plt.xlim(-1,1)
    plt.ylim(0,1)
    plt.xlabel('x')
    plt.ylabel('f(x),p(x)')
    plt.title('Problem 1')
    plt.show()
    
    # Problem 2
    p_lagrange = lagrange(x,y,N,a,b)
    
    plt.figure()
    plt.plot(x,y,'o')
    plt.plot(x,p_lagrange)
    plt.xlim(-1,1)
    plt.ylim(0,1)
    plt.xlabel('x')
    plt.ylabel('f(x),p(x)')
    plt.title('Problem 2')
    plt.show()
    
    # Problem 3
    for j in range(0,N): 
        x[j] = np.cos(((2*j - 1)*np.pi)/(2*N))
        
    p_lagrange = lagrange(x,y,N,a,b)
    
    plt.figure()
    plt.plot(x,y,'o')
    plt.plot(x,p_lagrange)
    plt.xlim(-1,1)
    plt.ylim(0,1)
    plt.xlabel('x')
    plt.ylabel('f(x),p(x)')
    plt.title('Problem 3')
    plt.show()

def coeff(x, y, N, a, b):
    V = np.zeros((N,N))
    c = np.zeros(N)
    xeval = np.linspace(a,b,N)
    
    V[:,0] = 1
    for j in range(1,N):
        V[:,j] = x**(j)
    
    c = inv(V)*y
    
    xvect = np.zeros(N)
    for x in range(0,N): 
        for j in range(0,N): 
            xvect[j] = float("%.8f"%(xeval[x])**j)
        p = c*xvect
    
    return c, p
    
def lagrange(x,y,N,a,b):
    xeval = np.linspace(a,b,N)
    p = np.zeros(N)
    prod = 1
    s = 0
    
    for c in range(0,N): 
        phi_n = 1
        for i in range(0,N): 
            phi = xeval[c]-x[i]
            if phi == 0: 
                phi_n = phi_n
            else:
                phi_n = phi_n*phi
            for j in range(0,N): 
                if i == j: 
                    wj = 0
                else: 
                    prod_temp = x[j]-x[i]
                    if prod_temp == 0: 
                        prod = prod
                    else: 
                        prod = float("%.10f"%(prod_temp*prod))
                    if prod == 0: 
                        wj = wj
                    else: 
                        wj = 1/prod
                if xeval[c]-x[j] == 0 or wj == 0 or y[j] == 0:
                    s = s
                else: 
                    sj = (wj*y[j])/(xeval[c]-x[j])
                    s = s + sj
        p[c] = phi_n*s

    return p
    
driver()
