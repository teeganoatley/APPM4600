import mypkg.my2DPlotB
import numpy as np
import math
from numpy.linalg import inv 
import matplotlib.pyplot as plt

def driver():
    
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex(j) == f(xeval(j)) 
      
    plt = mypkg.my2DPlotB(xeval,fex)
    plt.addPlot(xeval,yeval)
    plt.show()   
     
     
    err = abs(yeval-fex)
    plt2 = mypkg.my2DPlotB(xeval,err)
    plt2.show()            


    
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint(jint)
        fa1 = f(a1)
        b1 = xint(jint+1)
        fb1 = f(b1)
        
        for kk in range(n):
           '''use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
           point = line_eval(fa1,fb1,a1,b1,kk)
           
def line_eval(y0,y1,x0,x1,alpha): 
    y0 = f(x0)
    y1 = f(x1)
    
    m = (y1-y0)/(x1-x0)
    
    y = lambda x: m*(x-x0)+y0
    
    point = y(alpha)
    
    return point
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
