import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp,quad
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import erf

omega=1*(1e6)*2*np.pi
omegas=1.5*omega
alphas=0.1

tpi=np.pi/omega
def feq(t,y):
    a=y[0]
    b=y[1]
    
    derivs=[0.5*1j*omega*(1+2*alphas*np.cos(omegas*t))*b,
            0.5*1j*omega*(1+2*alphas*np.cos(omegas*t))*a]
    return derivs

def jac(t,y):
    a=y[0]
    b=y[1]
    return [[0,0.5*1j*omega*(1+2*alphas*np.cos(omegas*t))],
            [0.5*1j*omega*(1+2*alphas*np.cos(omegas*t)),0]]

    #intitial condition
y0=[1+0.0j,0+0.0j]
t0=[0,2*tpi]

sol = solve_ivp(feq,t0,y0,rtol=1e-7,atol=3e-7)
yf=(sol.y[0][-1],sol.y[1][-1])
    #print(yf)
res=[(abs(yf[0]))**2,(abs(yf[1]))**2]

yt=abs(sol.y[0])**2
xt=np.linspace(0,6,len(yt))
print(yt)
print(xt)

##plt.plot(xt,yt)
##plt.xlabel("t/t_π")
##plt.ylabel("population")
##plt.title("Omega=1MHz, Omegas=1.5MHz, α_s=0.7")
##
##plt.show()


print(res)

xa=[]
ya=[]
for i in range(200):
    omegas=(i+1)*2*omega/100
    print("--------------------")
    print(omegas/omega)
    sol = solve_ivp(feq,t0,y0,rtol=1e-7,atol=3e-7)
    yf=(sol.y[0][-1],sol.y[1][-1])
    #print(yf)
    res=[(abs(yf[0]))**2,(abs(yf[1]))**2]
    print(res)
    xa.append(omegas/omega)
    ya.append(res[1])

plt.plot(xa,ya)
plt.xlim(0,4)
plt.xlabel("Omega_s/Omgea")
plt.ylabel("Error")
plt.title("Pi pulse error, α_s=0.1")
plt.show()
    
