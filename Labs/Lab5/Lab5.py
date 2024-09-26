# import libraries
import numpy as np

def driver():

# use routines    
    #f = lambda x: ((x-1)**2)*(x-3)
    #a = 0
    #b = 2

    f = lambda x: np.exp(x**2+7*x-30)-1
    fp = lambda x: (np.exp(x**2+7*x-30))*(2*x+7)
    fpp = lambda x: (np.exp(x**2+7*x-30))*(4*x**2+28*x+51)
    a = 2
    b = 4.5

    tol = 1e-10

    [astar,ier] = bisection(f,fp,fpp,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))




# define routines
def bisection(f,fp,fpp,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    gp = (f(d)*fp(d))/(fpp(d)**2)
    while (gp<1):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      gp = (f(d)*fp(d))/(fpp(d)**2)
      count = count +1
      print(count)
#      print('abs(d-a) = ', abs(d-a))

    Nmax = 1000  
    [p,pstar,info,it] = newton(f,fp,d,tol,Nmax)
    
    astar = pstar
    ier = info
    return [astar, ier]
    
def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]
      
driver() 