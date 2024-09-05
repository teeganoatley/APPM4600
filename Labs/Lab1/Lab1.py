# importing libraries to use later
import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3]

X = np.linspace(0, 2 * np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)

#plotting the functions above
plt.plot(X, Ya)
plt.plot(X, Yb)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Exersizes 

#making 2 equal arrays
x = np.linspace(0, 10, 10)
y = np.arange(10, 20, 1)

first3x = x[:3]
print('the first three entries of x are', first3x)

w = 10**(-np.linspace(1,10,10))
x = np.linspace(0,len(w),10)
s = 3*w

plt.semilogy(x,w)
plt.semilogy(x,s)
plt.xlabel('x')
plt.ylabel('w, s')
plt.show()