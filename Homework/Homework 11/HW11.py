# Homework 11
import numpy as np
import math
from scipy.integrate import quad as quad

def driver():

    '''Problem 1'''
    print("Problem 1 Results")
    f = lambda s: 1/(1+s**2)
    N_trap = 1550 # from part b calculations
    N_simp = 712
    ''' interval'''
    a = -5
    b = 5
   
    #create equispaced points
    t_trap = np.linspace(a,b,N_trap+1)
    t_simp = np.linspace(a,b,N_simp+1)
    
    # part a
    trap_sum = trap(f,t_trap,N_trap)
    print("Composite Trapezoid Integral: ", trap_sum)
    
    simp_sum = trap(f,t_simp,N_simp)
    print("Composite Simpson's Rule Integral: ", simp_sum)
    
    # part c
    scipy = quad(f,a,b)
    print("SCIPY quad w/ default tol: ", scipy[0])
    
    scipy = quad(f,a,b,epsabs=10**-4)
    print("SCIPY quad w/ 10^-4 tol: ", scipy[0])
    print("\n")
    
    '''Problem 2'''
    f = lambda t: t*np.cos(1/t)
    a = 0
    b = 1
    N = 4
    
    nodes = np.linspace(a,b,N+1)
    
    comp =((nodes[1]-nodes[0])/(3*N))*(f(nodes[N]))

    for i in range(1,N-1):
        if nodes[i-1] == 0:
            s_trap = ((nodes[i])/(3*N))*(2*f(nodes[i]))
        else:
            s_trap = ((nodes[i] - nodes[i-1])/(3*N))*(4*f(nodes[i-1])+2*f(nodes[i]))
        comp += s_trap
        
    print("Problem 2 Results")
    print("Problem 2 Composite Simpson: ", comp)
    
    
def trap(f,t,N):
    comp = 0

    for i in range(1,N-1):
        a_trap =((t[i] - t[i-1])/2)*(f(t[i-1])+f(t[i]))
        comp += a_trap
    
    return comp
    
def simp(f,t,N):
    comp =((t[1]-t[0])/(3*N))*(f(t[0]) +f(t[N+1]))

    for i in range(1,N-1):
        s_trap =((t[i] - t[i-1])/(3*N))*(4*f(t[i-1])+2*f(t[i]))
        comp += s_trap
    
    return comp
    
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver() 