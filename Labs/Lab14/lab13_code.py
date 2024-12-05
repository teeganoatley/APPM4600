import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
from scipy.linalg import lu_factor, lu_solve
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 5000
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)
    
     start = time.time()
     x = scila.solve(A,b)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     print(r)
     
     stop = time.time()
     time_lab13 = stop - start
     
     print("Lab 13 Method Time = ", time_lab13)
     
     '''Lab 14 Part 1'''
     start = time.time()
     [lu,piv] = lu_factor(A)
     stop = time.time()
     time_lu_fact = stop - start
     
     start = time.time()
     x = lu_solve((lu,piv),b) #solution of square matrix
     stop = time.time()
     time_lu_solve = stop - start
     
     print("LU Factorization Time = ", time_lu_fact)
     print("LU Solve Time = ", time_lu_solve)
     
     ''' Create an ill-conditioned rectangular matrix '''
     N = 100
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
