#test

import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import complex_ode
from scipy.integrate import ode

#number of runs for averaging
nrun=10

#rabi parameters
omega_0=2*math.pi*(0.1)*(10e6)
ntime=1
timeupperbound=0.5*(ntime+5)*2*math.pi/(omega_0)
tpi=math.pi/(omega_0)
#frequency domain sample parameters
fmax=10*omega_0/(2*math.pi)
nf=10000
fmin=fmax/nf
df=fmin

#make time array for solution
ntimesweep=1000
tstop=timeupperbound
tinc=tstop/ntimesweep
tarray=np.arange(0.,tstop,tinc)

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwmax=0.3*omega_0/(2*math.pi)
nsweep=10
lwmin=lwmax/nsweep
dlw=lwmin

#the power spectral density of phase noise
def s_nu(f,lw):
    return lw/(2*f*f*math.pi)

#pre-calculating some constants
s_nu_arr=[]
pikdf_arr=[]
for k in range(nf):
    s_nu_arr.append(2*math.sqrt(s_nu((k+1)*df,lwmax)*df))
    pikdf_arr.append(2*math.pi*(k+1)*df)

#Arrays for recording the population of the 0 and 1 level after pi pulse for
#each run
y1run=[]
y0run=[]

for count in range(nrun):
    print("run #"+str(count)+"/"+str(nrun))

    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df. 
    phase_arr=[]
    for k in range(nf):
        phase_arr.append(random.uniform(0,2*math.pi))

    #function that returns the time-varying phasenoise
    def phigen(t):
        phitemp=0
        for k in range(nf):
            phitemp=phitemp+s_nu_arr[k]*math.cos(pikdf_arr[k]*t+phase_arr[k])
        return phitemp

    omega=omega_0
    #solving differential equations:defining equations and corresponding jacobians
    def feq(t,y):
        a=y[0]
        b=y[1]
        omega=omega_0
        derivs=[0.5*1j*omega*cmath.exp(1j*phigen(t))*b,
               0.5*1j*omega*cmath.exp(-1j*phigen(t))*a]
        return derivs
    def jac(t,y):
        a=y[0]
        b=y[1]
        return [[0,0.5*1j*omega*cmath.exp(1j*phigen(t))*b],
                [0.5*1j*omega*cmath.exp(1j*phigen(t))*b,0]]

    #intitial condition
    y0,t0=[1,0],0

    r=complex_ode(feq,jac).set_integrator('dopri5')
    r.set_initial_value(y0,t0)

    r.integrate(tpi)
    y0run.append((abs(r.y[0]))**2)
    y1run.append((abs(r.y[1]))**2)
#    while(r.successful() and r.t<timeupperbound):
#        r.integrate(r.t+tinc)

#        tsim.append(r.t)
#        asim.append((abs(r.y[0]))**2)
#        bsim.append((abs(r.y[1]))**2)

#    trun.append(tsim)
#    frun.append(asim)
print("Averaged populaton for each state:")
print("y[0]="+str(np.mean(y0run)))
print("y[1]="+str(np.mean(y1run)))

