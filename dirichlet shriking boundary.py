# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:02:51 2021

@author: sebzi
"""

import numpy as np
import matplotlib.pyplot as plt

import math

from mpl_toolkits.mplot3d import Axes3D

import time
from numba import njit

import matplotlib.animation as animation
import scipy.integrate as integrate

start = time.time()

#@njit(fastmath=True)
def soliton(x,t):

    w = -0.9995
    gamma = 1.0/((1.0-w**2.0)**(1.0/2.0))
    u = 4.0*np.arctan(np.exp(gamma*(x-w*t) ) )
    return u


#@njit(fastmath=True)
def dudt(x,t):

    w = -0.9995
    gamma = 1.0/((1.0-w**2.0)**(1.0/2.0))
    dudt = 4.0*(1.0 + np.exp(2.0*gamma*(x-w*t)) )**(-1.0)*gamma*w*np.exp(gamma*(x-w*t) )
    return dudt

#@njit(fastmath=True)
def sinG(L0,h, omega, T, timestep):
    
    t = np.arange(0, T, timestep)
    
    L = []
    interval = []
    k = timestep

    theta = 0.5
    
#    x = np.arange(-L0/2,L0/2+ 0.5*h, h)
    right = np.arange(0, L0/2+ 0.5*h,h)
    left = np.flip(-right[1:])
    x = np.concatenate((left,right))
    interval.append(x)
    
    u0 = np.zeros(len(x))
    v0 = np.zeros(len(x))
    for i in range(len(x)):
        u0[i] = soliton(x[i],0) #initial soliton
        v0[i] = dudt(x[i],0) #initial v
    
    u = []
    v = []
    
    u.append(np.zeros(len(x)))
    v.append(np.zeros(len(x)))
    u[0] = u0
    v[0] = v0                   
    
    

    for n in range(0,len(t)-1):
     #   print n
        a = []
        L = (L0*np.cos(omega*t[n]))

        right = np.arange(0, L/2 + 0.5*h,h)
        left = np.flip(-right[1:])
        x = np.concatenate((left,right))        
        #x = np.arange(-L/2,L/2+ 0.5*h, h) #changed this to centre on x
        
        
        

        R = np.zeros(len(x))
           
        M = len(x)
     #   print(M)
                 
        if n>0:
            interval.append(x)
            cutoff = int((len(interval[n-1]) - len(interval[n]))/2)
            
            u[n] = u[n][cutoff: len(u[n])- cutoff] #had plus 1 check this
            v[n] = v[n][cutoff: len(v[n])- cutoff]
        
        u.append(np.zeros(len(x)))
        v.append(np.zeros(len(x)))       
        
        
        for i in range(0,M):
            a.append(np.zeros(M))
    
        a[0][0] = 1.0
        a[0][1] = 0.0
    
        R[0] = 0
    
        for i in range(1, len(x)-1):
            a[i][i-1] = -theta/(h**2.0)
            a[i][i] = 1.0/(k**2.0 * theta) + theta*np.cos(u[n][i]) + 2.0*theta/(h**2.0)
         
            a[i][i+1] = -theta/(h**2.0)
    
            R[i] = v[n][i]/(k*theta) + 1.0/(h**2.0)*(u[n][i+1]-2*u[n][i] + u[n][i-1]) - np.sin(u[n][i])
    
       
    
        a[len(x)-1][len(x)-2] = 0
        a[len(x)-1][len(x)-1] = 1.0
    
        R[len(x)-1] = 0
    
        #now apply crute  factorisation
        # replace u in crute with w
    
        #creating empty arrays to fill
        L = []
        w = []
    
        for i in range(M):
            L.append(np.zeros(M))
            w.append(np.zeros(M))
    
        z = np.zeros(M)
        delta = np.zeros(M)
    
        L[0][0] = a[0][0] #L11
    
        w[0][1] = a[0][1]/L[0][0] #u12
    
        z[0] = R[0]/L[0][0] #z1
    
        for i in range(1, len(a[0])-1): #step 2
            L[i][i-1] = a[i][i-1]
            L[i][i] = a[i][i] - L[i][i-1]*w[i-1][i]
            w[i][i+1] = a[i][i+1]/L[i][i]
            z[i] = (R[i] - L[i][i-1]*z[i-1])/L[i][i]
    
    
        N = len(a[0])
    
        #step 3
        L[N-1][N-2] = a[N-1][N-2]
        L[N-1][N-1] = a[N-1][N-1] - L[N-1][N-2]*w[N-2][N-1]
        z[N-1] = (R[N-1] - L[N-1][N-2]*z[N-2])/L[N-1][N-1]
    
        delta[N-1] = z[N-1] #delta is entries solved for
                       
        #step 5
        for i in np.arange(N-2,-1,-1):
                 delta[i] = z[i] - w[i][i+1]*delta[i+1]
    
    
    
        #next time step
        for i in range(0,M):
            u[n+1][i] = delta[i] + u[n][i]
          #  print u[n+1][i]
            v[n+1][i] = delta[i]/(k*theta) - v[n][i]/theta + v[n][i]
    

        
    L = (L0*np.cos(omega*t[-1]))
        
#    x = np.arange(-L/2,L/2+ 0.5*h, h)
    interval.append(x)
    return t,interval,u


timestep = 0.001
res = sinG(1.0,0.001,0.5,2.0,timestep) #had omega = 0.5
t = res[0]
x = res[1]
u = res[2]   



Ep = [] #potential energy
Es = []
Ek = []
Etot = []

#Ek calculation
g = []
for i in range(len(u)):
    c = int((len(u[0]) - len(u[i]))/2) #difference in arrays
    d = np.full(c,'q')
    e = np.concatenate((u[i],d))

    f= np.concatenate((d,e))  #adding qs to both ends of the array

    g.append(f)
gt = np.transpose(g) #transposed so gives all time values for each x point
propergt=[]
for i in range(len(gt)):
    a = np.delete(gt[i], np.where(gt[i]=='q'))  #deleting all the qs I've added
    propergt.append(a.astype('float')) #chamging from strings to floats

grad = []
for i in range(len(propergt)):
    grad.append(np.gradient(propergt[i],timestep)) #finding the time gradient for each x point

#final = np.transpose(grad)
#take the gradient and then invert again
h = len(grad[int(len(grad)/2)]) #taking length of the biigest time position array i.e. middle one 
toinvert =[]
for i in range(len(grad)):
    j = h- len(grad[i])
    k = np.full(j, 'q')
    l = np.concatenate((grad[i],k))
    toinvert.append(l)
#final = np.transpose(toinvert)
remove = np.transpose(toinvert) #now need to get rid of zeros and then integrate
final = []
for i in range(len(remove)):
    a = np.delete(remove[i], np.where(remove[i]=='q'))
    final.append(a.astype('float'))
    
    
 
for i in range(len(u)-1):
#    P = 0
#    for j in range(len(x)):
#        P += (1-np.cos(u[i][j]))*h 
#    Ep.append(P)
   
    Ep.append(integrate.simps(1.0-np.cos(u[i]),x[i]))
    Es.append(0.5*integrate.simps( (np.gradient(u[i],x[i]))**2.0, x[i]  ) )
    Ek.append(0.5*integrate.simps(final[i]**2,x[i]))
    Etot.append(Ep[i]+Es[i]+Ek[i])


plt.figure()
plt.plot(t[0:len(t)-1],Ep, label = 'Ep')
plt.plot(t[0:len(t)-1],Es, label = 'Es')
plt.plot(t[0:len(t)-1],Ek, label = 'Ek')
plt.plot(t[0:len(t)-1],Etot, label = 'Etot')
plt.legend()