# Homework 10
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():

    # Problem 1
    f = lambda x: np.sin(x)
    partA = lambda x: (x - (7/60)*x**3)/(1 + (1/20)*x**2)
    partB = lambda x: x / (1 + (1/6)*x**2 + (13/360)*x**4)
    partC = lambda x: (x - (7/60)*x**3)/(1 + (1/20)*x**2)
    Mpoly = lambda x: x - (1/6)*x**3 + (1/120)*x**5

    N = 100
    ''' interval'''
    a = 0
    b = 5
   
    ''' create equispaced interpolation nodes for Question 1'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = np.zeros(N+1)
    Ay = np.zeros(N+1)
    By = np.zeros(N+1)
    Cy = np.zeros(N+1)
    My = np.zeros(N+1)
    for jj in range(N+1):
        yint[jj] = f(xint[jj])
        Ay[jj] = partA(xint[jj])
        By[jj] = partB(xint[jj])
        Cy[jj] = partC(xint[jj])
        My[jj] = Mpoly(xint[jj])
    
    
    plt.figure()
    plt.plot(xint,yint,'ro-',label='f(x)')
    plt.plot(xint,Ay,'bs--',label='Part A Pade') 
    plt.plot(xint,By,'c.--',label='Part B Pade')
    plt.plot(xint,Cy,'g.-',label="Part C Pade")
    plt.plot(xint,My,'m.-.',label="Maclaurin")
    plt.title("Approximation Plots")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim((a,b))
    plt.legend()
    plt.semilogy()
    plt.show()
         
    errA = abs(Ay-My)
    errB = abs(By-My)
    errC = abs(Cy-My)

    plt.figure()
    plt.semilogy(xint,errA,'bs--',label='Part A Error')
    plt.semilogy(xint,errB,'c.--',label='Part B Error')
    plt.plot(xint,errC,'g.-',label="Part C Error")
    plt.title("Absolute Error")
    plt.xlim((a,b))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()  

    
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver() 