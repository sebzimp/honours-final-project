# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:56:27 2021

@author: sebzi
"""

import numpy as np
import matplotlib.pyplot as plt

import math

from mpl_toolkits.mplot3d import Axes3D

import time
from numba import njit

import matplotlib.animation as animation

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate

start = time.time()

@njit(fastmath=True)
def soliton(x,t):

    w = -0.9995
    gamma = 1.0/((1.0-w**2.0)**(1.0/2.0))
    u = 4.0*np.arctan(np.exp(gamma*(x-w*t) ) )
    return u


@njit(fastmath=True)
def dudt(x,t):

    w = -0.9995
    gamma = 1.0/((1.0-w**2.0)**(1.0/2.0))
    dudt = 4.0*(1.0 + np.exp(2.0*gamma*(x-w*t)) )**(-1.0)*gamma*w*np.exp(gamma*(x-w*t) )
    return dudt

@njit(fastmath=True)
def sinG(xL,xR,xstep, T, timestep):
    x = np.arange(xL,xR+ 0.5*xstep, xstep)
    t = np.arange(0, T, timestep)
    
    k =timestep
    h =  xstep
    theta = 0.5
    
    u0 = np.zeros(len(x))
    v0 = np.zeros(len(x))
    for i in range(len(x)):
        u0[i] = soliton(x[i],0) #initial soliton
        v0[i] = dudt(x[i],0) #initial v
    
    u = []
    v = []
    
    for i in range(len(t)):
        u.append(np.zeros(len(x)))
        v.append(np.zeros(len(x)))
    
    R = np.zeros(len(x))
    u[0] = u0
    v[0] = v0
                 
    M = len(x)

    
    for n in range(0,len(t)-1):
     #   print n
        a = []
    
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
    
       
    
        a[len(x)-1][len(x)-2] =  0.0
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
    
#        c = c + delta[1]
#        d = d + delta[-2]
    return t,x,u

res = sinG(-0.5,0.5,0.001,4,0.001)
t = res[0]
x = res[1]
u = res[2]

#energy stuff
u = np.array(u)
ut = np.transpose(u)


#energy error calculations
ddt = []
Ep = [] #potential energy
Es = []
Ek = []
Etot = []

for i in range(len(ut)):
    ddt.append(np.gradient(ut[i],t)) #du/dt
right = np.transpose(ddt)
 
for i in range(len(u)):
    P = 0
#    for j in range(len(x)):
#        P += (1-np.cos(u[i][j]))*h 
#    Ep.append(P)
    Ep.append(integrate.simps(1.0-np.cos(u[i]),x))
    Es.append(0.5*integrate.simps( (np.gradient(u[i],x))**2.0, x  ) )
    Ek.append(0.5*integrate.simps(right[i]**2,x))
    Etot.append(Ep[i]+Es[i]+Ek[i])

plt.figure()
plt.plot(t,Ek, label = 'Ek')   
plt.plot(t,Es, label = 'Es')   
plt.plot(t,Ep, label = 'Ep')   
plt.plot(t,Etot, label = 'Tot E') 
plt.legend()
end = time.time()

print(end - start)

#fig, ax = plt.subplots(figsize=(5, 3))
#ax.set(xlim=(-0.6, .6), ylim = (-20,10))

#X2, T2 = np.meshgrid(x, t)

#line = ax.plot(x, u[0], color='k', lw=2)[0]
#def animate(i):
#    line.set_ydata(u[i*8])

#anim = animation.FuncAnimation(fig, animate, interval=10, frames=len(t)-1)

#plt.draw()
#plt.show()