import math
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv 

def driver():
    
    #Problem 2 
    A = np.array([[12, 10, 4],[10, 8, -5],[4, -5, 3]])
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    alpha = -1*((A[1][0])**2 + (A[2][0])**2)**(1/2)
    r = (((alpha**2)/2) - 0.5*(A[1][0])*alpha)**(1/2)
    
    w1 = 0
    w2 = (A[1][0]-alpha)/(2*r)
    w3 = A[2][0]/(2*r)
    
    w = np.array([[w1], [w2], [w3]])
    
    W = w@w.transpose()
    
    H = I - 2*W
    
    HA = H@A
    A_td = HA@H
    print(A_td)
    print("")
    
    #Problem 3 Part a 
    n = np.linspace(4,20,5)
    N = 1000
    tol = 1e-4
    
    for ii in range(0,len(n)):
        A = hilbert(int(n[ii]))
        x0 = np.ones(int(n[ii]))
        [eig_val, eig_vec, count] = power(A,x0,tol,N)
        print("Corresponding eigenvalue: ", eig_val)
        print("Number of iterations: ", count)
        print("")
        
    #Problem 3 Part b 
    n = np.linspace(4,20,5)
    N = 1000
    tol = 1e-4
    ''' getting a singular matrix error?
    for ii in range(0,len(n)):
        A = inv(hilbert(int(n[ii])))
        x0 = np.ones(int(n[ii]))
        [eig_val, eig_vec, count] = power(A,x0,tol,N)
        eig_val = 1/eig_val
        print("Corresponding eigenvalue: ", eig_val)
        print("Number of iterations: ", count)
        print("")
    '''
    
    #Problem 3 Part c
    
    
    
    
def power(A,x0,tol,max_iter):
    eig_vec = x0
    count = 0
    m1 = 1
    
    y = A@eig_vec
    eig_val = max(abs(y))
    
    err = abs(m1-eig_val)
    
    while err > tol:
        y = A@eig_vec
        eig_val = max(abs(y))
        eig_vec = y/eig_val
        err = abs(m1-eig_val)
        count = count + 1
        m1 = eig_val
            
    return eig_val, eig_vec, count
    
def hilbert(N):

    A = np.zeros((N,N))

    for i in range(1,N):
        for j in range(1,N):
            A[i][j] = 1/(i+j-1)
    
    return A      
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
