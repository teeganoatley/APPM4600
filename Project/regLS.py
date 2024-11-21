import numpy as np
import math
import matplotlib.pyplot as plt

def driver(): 

    '''define variables'''
    N = 100
    a = -5
    b = 5
    xint = np.linspace(a,b,N+1)
    n_mag = 5 # noise magnitude
    np.random.seed(10)
    

    '''define functions for experiments'''
    f1 = lambda x: 2*x-1 # linear
    f2 = lambda x: 2*x**2 + 2*x - 1 # quadratic 
    f3 = lambda x: -1*math.exp(x-1) # exponential 
    f4 = lambda x: 4*math.sin(6*x) # sinusoidal
    
    # create y-vectors from functions 
    yint1 = np.zeros(N+1)
    yint2 = np.zeros(N+1)
    yint3 = np.zeros(N+1)
    yint4 = np.zeros(N+1)
    for jj in range(N+1):
        yint1[jj] = f1(xint[jj])
        yint2[jj] = f2(xint[jj])
        yint3[jj] = f3(xint[jj])
        yint4[jj] = f4(xint[jj])

    yint1_n = noise(yint1,N,n_mag)
    yint2_n = noise(yint2,N,n_mag)
    yint3_n = noise(yint3,N,n_mag)
    yint4_n = noise(yint4,N,n_mag)
    
    '''plot for each function (clean and with noise)'''
    # f1
    plt.figure()
    plt.plot(xint,yint1_n,'bo--',label='Noisy Function')
    plt.plot(xint,yint1,'r-',label='Real Function')
    plt.title("f(x)=2x-1, N=100")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim((a,b))
    plt.legend()
    plt.show()
    
    # f2
    plt.figure()
    plt.plot(xint,yint2_n,'bo--',label='Noisy Function')
    plt.plot(xint,yint2,'r-',label='Real Function')
    plt.title("f(x)=2x^2+2x-1, N=100")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim((a,b))
    plt.legend()
    plt.show()
    
    # f3
    plt.figure()
    plt.plot(xint,yint3_n,'bo--',label='Noisy Function')
    plt.plot(xint,yint3,'r-',label='Real Function')
    plt.title("f(x)=-exp(x-1), N=100")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim((a,b))
    plt.legend()
    plt.show()
    
    # f4
    plt.figure()
    plt.plot(xint,yint4_n,'bo--',label='Noisy Function')
    plt.plot(xint,yint4,'r-',label='Real Function')
    plt.title("f(x)=4sin(6x), N=100")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim((a,b))
    plt.legend()
    plt.show()


def noise(yint,N,n_mag):
    mean = 0 # required for functional integrity 
    std = n_mag
    noise = np.random.normal(mean,std,N+1) # normal distribution of noise

    yint = yint + noise
    
    return yint

def LS(xint,yint,N,a,b):

    return

if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver() 
