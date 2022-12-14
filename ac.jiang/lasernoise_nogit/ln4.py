#test
from joblib import Parallel, delayed #for parallel run
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import complex_ode
from scipy.integrate import ode

#number of runs for averaging
nrun=2000

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
lwmax=0.3*omega_0/(2*np.pi)
nsweep=10
lwmin=lwmax/nsweep
dlw=lwmin

#the power spectral density of phase noise
def s_nu(f,lw):
    return lw/(2*f*f*math.pi)

#pre-calculating some constants
s_nu_arr= np.zeros(nf)
pikdf_arr= np.zeros(nf)
for k in range(nf):
    s_nu_arr[k] = 2*np.sqrt(s_nu((k+1)*df,lwmax)*df)
    pikdf_arr[k] = 2*np.pi*(k+1)*df

#Arrays for recording the population of the 0 and 1 level after pi pulse for
#each run
y1run=[]
y0run=[]
np.random.seed(0)
#function that returns the time-varying phasenoise
def phigen(t,phase_arr):
    phitemp = np.sum(s_nu_arr*np.cos(pikdf_arr*t+phase_arr))
    return phitemp


def run_job():
    #print("run #"+str(count)+"/"+str(nrun))

    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df.
    phase_arr=np.random.uniform(0,2*np.pi,size=nf)

    omega=omega_0
    def feq(t,y):
        a=y[0]
        b=y[1]
        omega=omega_0
        derivs=[0.5*1j*omega*np.exp(1j*phigen(t,phase_arr))*b,
                0.5*1j*omega*np.exp(-1j*phigen(t,phase_arr))*a]
        return derivs

    def jac(t,y):
        a=y[0]
        b=y[1]
        return [[0,0.5*1j*omega*np.exp(1j*phigen(t,phase_arr))*b],
                [0.5*1j*omega*np.exp(1j*phigen(t,phase_arr))*b,0]]

    #intitial condition
    y0,t0=[1+0.0j,0+0.0j],0

    r=complex_ode(feq,jac).set_integrator('dopri5')
    r.set_initial_value(y0,t0)

    r.integrate(tpi)
    # y0run.append((abs(r.y[0]))**2)
    # y1run.append((abs(r.y[1]))**2)
    return [(abs(r.y[0]))**2,(abs(r.y[1]))**2]
    # y0run.append(
    # y1run.append((abs(r.y[1]))**2)

# results = []
# for count in range(nrun):
#     results.append(run_job())
# results = np.array(results)
#print(results)

#For parallel use below
num_cores = 1
results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
results = np.array(results)
print("Averaged population for each state:")
print("y[0]="+str(np.mean(results[:,0])) + "+-" + str(np.std(results[:,0])/np.sqrt(nrun)))
print("y[1]="+str(np.mean(results[:,1]))+ "+-" + str(np.std(results[:,1])/np.sqrt(nrun)))

