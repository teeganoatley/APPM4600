import math
import numpy as np
import matplotlib.pyplot as plt
import random

def driver(): 

    # Problem 3
    x = 9.999999995000000 * 10**(-10)

    y = math.e**x
    f_taylor = x + (x**2)/2 + (x**3)/6 + (x**4)/24
    
    print("basic algorithm: ", y-1)
    print("taylor series: ", f_taylor)
    
    # Problem 4 Part a
    t = np.arange(0,np.pi,(np.pi)/30)
    y = np.cos(t)
    k = 1
    S = 0
    
    for k in range(0, len(t)):
        S = S + t[k]*y[k]
    
    print("the sum is: ",S)
    
    # Problem 4 part b
    R = 1.2
    delta_r = 0.1
    f = 15
    p = 0
    theta = np.linspace(0,2*np.pi,100)
    
    x = R*(1+delta_r*np.sin(f*theta + p))*np.cos(theta)
    y = R*(1+delta_r*np.sin(f*theta + p))*np.sin(theta)
    
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.show()
    
    for i in range(0, 10):
        R = i
        delta_r = 0.05
        f = 2+i
        p = random.uniform(0,2)
        
        x = R*(1+delta_r*np.sin(f*theta + p))*np.cos(theta)
        y = R*(1+delta_r*np.sin(f*theta + p))*np.sin(theta)
        
        plt.plot(x,y)
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-10.10)
    plt.ylim(-10.10)
    plt.show()
    
    return
    
driver()