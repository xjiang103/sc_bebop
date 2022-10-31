import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import complex_ode
from scipy.integrate import ode

#number of runs
nrun=10



#rabi parameters
omega_0=2*math.pi*(0.101528)*(10e6)
ntime=1
timeupperbound=0.5*(ntime+5)*2*math.pi/(omega_0)

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

#linewidth sweep parameters
lwmax=0.3*omega_0/(2*math.pi)
nsweep=10
lwmin=lwmax/nsweep
dlw=lwmin

def s_nu(f,lw):
    return lw/(2*f*f*math.pi)


print(s_nu(100,100))

y1run=[]
y0run=[]
for count in range(nrun):
    print(count)
    timet=1000*time.time()
    seedarray=[]
    for k in range(nf):
        random.seed()   
        seedarray.append(random.randint(1,10000000))
    
    def phigen(t):
        phitemp=0
        for k in range(nf):
            random.seed(seedarray[k])
            phitemp=phitemp+2*math.sqrt(s_nu((k+1)*df,lwmax)*df)*math.cos(2*math.pi*(k+1)*df*t+random.uniform(0,2*math.pi))
        return phitemp
#phiarray=[]
#for k in range(ntimesweep+1):
#    phiarray.append(phigen(tarray[k]))

#plt.plot(tarray,phiarray)
#plt.show()
#_------------------------------------------------------------------------
    omega=omega_0
#solving differential equations
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

    y0,t0=[1,0],0

    r=ode(feq,jac).set_integrator('zvode',method='bdf')
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

print("y[0]="+str(np.mean(y0run)))
print("y[1]="+str(np.mean(y1run)))

