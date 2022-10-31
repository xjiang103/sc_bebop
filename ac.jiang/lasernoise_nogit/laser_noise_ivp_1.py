import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy import stats
nrun=10
units=1e0
#rabi parameters
omega1=2*np.pi*(10)*(units)
omega2=2*np.pi*(5)*(units)
delta1=2*np.pi*(246.2)*(units)
delta2=delta1*(1-np.sqrt(1+(omega1**2-omega2**2)/(2*delta1**2)))-delta1
delta=delta1-delta2
deltasum=delta1+delta2
delta1prime=deltasum+(omega1**2)/delta
delta2prime=deltasum-(omega2**2)/delta
deltaplus=(delta1prime+delta2prime)/2
deltaminus=(delta1prime-delta2prime)/2
omegar=omega1*omega2/delta
omegaprime=np.sqrt(omegar**2+deltaplus**2)
print("δ="+str(delta/(2*np.pi*units)))
print("ΩR="+str(omegar/(2*np.pi*units)))
print("Ωprime="+str(omegaprime/(2*np.pi*units)))
print(delta)
print(omega1)
print(omega2)

tpi=np.pi/omegar
print(tpi)

#frequency domain sample parameters
fmax=10*omegar/(2*np.pi)
nf=10000
fmin=fmax/nf
df=fmin
#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwmax=0.3*omegar/(2*np.pi)
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
np.random.seed()
#function that returns the time-varying phasenoise
def phigen1(t,phase_arr):
    phitemp = np.sum(s_nu_arr1*np.cos(pikdf_arr*t+phase_arr))
    return phitemp

def phigen2(t,phase_arr):
    phitemp = np.sum(s_nu_arr2*np.cos(pikdf_arr*t+phase_arr))
    return phitemp

def s_nu(f,lw):
    return lw/(2*f*f*math.pi)

def run_job():
    #print("run #"+str(count)+"/"+str(nrun)) 
    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df.
    phase_arr1=np.random.uniform(0,2*np.pi,size=nf)
    phase_arr2=np.random.uniform(0,2*np.pi,size=nf)

    def feq(t,y):
        a=y[0]
        b=y[1]
        c=y[2]
        phi1 =phigen1(t,phase_arr1)
        phi2 =phigen2(t,phase_arr2)
        derivs=[0.5*1j*omega1*np.exp(1j*(delta1*t+phi1))*b,
                0.5*1j*omega1*np.exp(-1j*(delta1*t+phi1))*a+0.5*1j*omega2*np.exp(1j*(delta2*t+phi2))*c,
                0.5*1j*omega2*np.exp(-1j*(delta2*t+phi2))*b]

        # derivs=[0.5*1j*omega1*np.exp(1j*(phigen1(t,phase_arr1)))*b,
        #         0.5*1j*omega1*np.exp(-1j*(phigen1(t,phase_arr1)))*a+0.5*1j*omega2*np.exp(1j*(phigen2(t,phase_arr2)))*c,
        #         0.5*1j*omega2*np.exp(-1j*(phigen2(t,phase_arr2)))*b]

        # derivs=[0.5*1j*omega1*np.exp(1j*(delta1*t))*b,
        #         0.5*1j*omega1*np.exp(-1j*(delta1*t))*a+0.5*1j*omega2*np.exp(1j*(delta2*t))*c,
        #         0.5*1j*omega2*np.exp(-1j*(delta2*t))*b]

        return derivs

    # def jac(t,y):
    #     return [[0,0.5*1j*omega1*np.exp(1j*(delta1*t)),0],
    #             [0.5*1j*omega1*np.exp(-1j*(delta1*t)),0,0.5*1j*omega2*np.exp(1j*(delta2*t))],
    #             [0,0.5*1j*omega2*np.exp(-1j*(delta2*t)),0]]
    #intitial condition
    y0=[1+0.0j,0+0.0j,0+0.0j]
    t = [0,tpi]

    sol = solve_ivp(feq,t,y0,rtol=1e-7,atol=3e-7)
    yf = [sol.y[0][-1],sol.y[1][-1],sol.y[2][-1]]
    return [(abs(yf[0]))**2,(abs(yf[1]))**2,(abs(yf[2]))**2]




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
    return [1-meantotresult[2],(np.std(totresult,axis=0))[2]]


nsweep = 5
nrun = 5

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
