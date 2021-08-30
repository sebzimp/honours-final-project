# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:31:34 2021

@author: sebzi
"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure(dpi = 200)
plt.xlabel("x")
plt.ylabel("u(x)")

for i in range(1,10):
    sol = np.loadtxt("L1.0" + str(i)+"after10.txt")
    x = np.linspace(-0.5 - i*0.01/2,0.5 + i*0.01/2, 800 + i*8)
    
    plt.plot(x,sol, label = "L = 1.0" + str(i))
        
 #   a = np.gradient(sol, x)
plt.legend(loc = "upper right")  
#    centre  = np.where(abs(a)==max(abs(a)))
    
#    print(x[centre])
#    print(sol[centre])


#sol2 = np.loadtxt("L0.91after10.txt")
#sol3 = np.loadtxt("L1.1after10.txt")

#x = np.linspace(-0.455,0.455,728)

#a = np.gradient(sol2, x)

#centre  = np.where(abs(a)==max(abs(a)))
#print(centre)
#print(x[centre])
#print(sol2[centre])
#x2 = np.linspace(-0.45,0.45,720)
#x3 = np.linspace(-0.55,0.55,880)
#plt.plot(x,sol)
#plt.plot(x2,sol2)
#plt.plot(x3,sol3)