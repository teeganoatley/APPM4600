# import libraries
import numpy as np
import matplotlib.pyplot as plt
    
def driver():
#plot 
    x = np.linspace(-7*np.pi,7*np.pi,100)
    y = x-4*np.sin(x)-3

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.grid()
    plt.show()

# test functions 
    f = lambda x: -np.sin(2*x) + (5/4)*x - (3/4)

    Nmax = 100
    tol = 1e-10

# test f '''
    x0 = 8
    [xstar,ier] = fixedpt(f,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('f1(xstar):',f(xstar))
    print('Error message reads:',ier)



# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    

driver()