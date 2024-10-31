import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad as quad

def driver():

#  function you want to approximate
    f = lambda x: math.exp(x)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
      pval[kk] = eval_legendre(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()
    
      
def eval_legendre(f,a,b,w,n,x):
    phi = np.zeros(n+1)
    a = np.zeros(n+1)
    p = 0
    
    for i in range(0,n+1):
        if i == 0 or i== 1:
            phi[0] = 1
            phi[1] = x
        else:
            phi[i] = (1/i+1)*((2*n+1))*x*phi[i-1] - i*phi[i-2]
        
        num = lambda x: phi[i]*f(x)*w(x)
        denom = lambda x: ((phi[i])**2)*w(x)
        
        a_num, err = quad(num,-1,1)
        a_den, err = quad(denom,-1,1)
        
        if a_den == 0: 
            a[i] = 0
        else:
            a[i] = a_num/a_den
        
        p = p + a[i]*phi[i]
        
    return p
    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()               
