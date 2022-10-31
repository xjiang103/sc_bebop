#test
#test
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import complex_ode
from scipy.integrate import ode
from scipy import stats

#number of runs for averaging
nrun=30
xx=10e6
yy=1e6
print(xx-yy)
#rabi parameters
omega1=2*np.pi*(10)*(10e6)
omega2=2*np.pi*(5)*(10e6)
delta1=2*np.pi*(246.2)*(10e6)
delta2=delta1(1-np.sqrt(1+(omega1**2-omega2**2)/(2*delta1**2)))-delta1
delta=delta1-delta2
deltasum=delta1+delta2

delta1prime=deltasum+(omega1**2)/delta
delta2prime=deltasum-(omega2**2)/delta
deltaplus=(delta1prime+delta2prime)/2
deltaminus=(delta1prime-delta2prime)/2

omegar=omega1*omega2/delta
omegaprime=np.sqrt(omegar**2+deltaplus**2)

print("δ="+str(delta))
print("ΩR="+str(omegar))
print("Ωprime="+str(omegaprime))

tpi=np.pi/omegar

#frequency domain sample parameters
fmax=10*omega0/(2*np.pi)
nf=10000
fmin=fmax/nf
df=fmin

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwmax=0.3*omega_0/(2*np.pi)
nsweep=10
lwmin=lwmax/nsweep
dlw=lwmin
lwarray=np.arange(0.03,0.3,0.03)
#pre-calculating some constants

pikdf_arr= np.zeros(nf)
s_nu_arr1= np.zeros(nf)
s_nu_arr2= np.zeros(nf)

for k in range(nf):
    pikdf_arr[k] = 2*np.pi*(k+1)*df

#Arrays for recording the population of the 0 and 1 level after pi pulse for
#each run
y1run=[]
y0run=[]
np.random.seed()
#function that returns the time-varying phasenoise
def phigen1(t,phase_arr):
    phitemp = np.sum(s_nu_arr1*np.cos(pikdf_arr*t+phase_arr))
    return phitemp
def phigen2(t,phase_arr):
    phitemp = np.sum(s_nu_arr2*np.cos(pikdf_arr*t+phase_arr))
def s_nu(f,lw):
    return lw/(2*f*f*math.pi)

def run_job():
    #print("run #"+str(count)+"/"+str(nrun))

    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df.
    phase_arr1=np.random.uniform(0,2*np.pi,size=nf)
    phase_arr2=np.random.uniform(0,2*np.pi,size=nf)
    def feq(t,y):
        g=y[0]
        e=y[1]
        r=y[2]
        derivs=[0.5*1j*omegar*np.exp(1j*(delta1*t+phigen1(t,phase_arr1)))*e,
                0.5*1j*omegar*np.exp(-1j*(delta1*t+phigen1(t,phase_arr1)))*g+0.5*1j*omegar*np.exp(1j*(delta2*t+phigen2(t,phase_arr2)))*r,
                0.5*1j*omegar*np.exp(-1j*(delta2*t+phigen2(t,phase_arr2)))*e]
        return derivs

    def jac(t,y):
        g=y[0]
        e=y[1]
        r=y[2]
        return [[0,0.5*1j*omegar*np.exp(1j*(delta1*t+phigen1(t,phase_arr1)))*e,0],
                [0.5*1j*omegar*np.exp(-1j*(delta1*t+phigen1(t,phase_arr1)))*g,0,.5*1j*omegar*np.exp(1j*(delta2*t+phigen2(t,phase_arr2)))*r],
                [0,.5*1j*omegar*np.exp(-1j*(delta2*t+phigen2(t,phase_arr2)))*e,0]]

    #intitial condition
    y0,t0=[1+0.0j,0+0.0j,0+0.0j],0

    r=complex_ode(feq,jac).set_integrator('dopri5')
    r.set_initial_value(y0,t0)

    r.integrate(tpi)
    # y0run.append((abs(r.y[0]))**2)
    # y1run.append((abs(r.y[1]))**2)
    return [(abs(r.y[0]))**2,(abs(r.y[1]))**2,(abs(r.y[2]))**2]
    # y0run.append(
    # y1run.append((abs(r.y[1]))**2)
    
def swp_lw(lw1,lw2):
    for k in range(nf):
        s_nu_arr1[k] = 2*np.sqrt(s_nu((k+1)*df,lw1)*df)
        s_nu_arr2[k] = 2*np.sqrt(s_nu((k+1)*df,lw2)*df)
    res=[]
    for count in range(nrun):
        res.append(run_job())
    totresult=np.array(res)
    meantotresult=np.mean(totresult,axis=0)
    print(str(i)+str(meantotresult))
    return [1-meantotresult[2],(stats.sem(totresult,axis=0))[2]]

 
mean_arr=np.zeros(nsweep)
sem_arr=np.zeros(nsweep)
timet=time.time()
for i in range(nsweep):
    results = swp_lw((i+1)*dlw,(i+1)*dlw)
    mean_arr[i]=results[0]
    sem_arr[i]=results[1]
print(time.time()-timet)

print(lwarray)
print(mean_arr)
plt.errorbar(lwarray,mean_arr,xerr=0,yerr=sem_arr,ecolor='r')
plt.title('Error at π pulse, white noise, F_cutoff=10Ω0')
plt.xlabel('Linewidth/Ω0')
plt.ylabel('Error')
plt.show()


#For parallel use below
##num_cores = 1
##results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
##results = np.array(results)
##print("Averaged population for each state:")
##print("y[0]="+str(np.mean(results[:,0])) + "+-" + str(np.std(results[:,0])/np.sqrt(nrun)))
##print("y[1]="+str(np.mean(results[:,1]))+ "+-" + str(np.std(results[:,1])/np.sqrt(nrun)))
##
