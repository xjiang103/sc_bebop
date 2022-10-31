
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy import stats

#number of runs for averaging
nrun=100

#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=1*math.pi/(omega_0)
tpi=tpi0
t2pi=2*tpi0
#frequency domain sample parameters
hwhm=omega_0/(20*np.pi)
fmax=4*hwhm
nf=1000
fmin=fmax/nf
df=fmin
delta=0
#pre-calculating some constants

pikdf_arr= np.zeros(nf)
s_nu_arr= np.zeros(nf) 
for k in range(nf):
    pikdf_arr[k] = np.pi*(k+1)*df

np.random.seed()
#function that returns the time-varying phasenoise
def phigen(t,phase_arr):
    phitemp = np.sum(s_nu_arr*np.cos(pikdf_arr*t+phase_arr))
    return phitemp
def s_f(f,sigma):
    #return (sigma**2*np.sqrt(2.0/np.pi)*hwhm/(hwhm*2+f*2))
    return sigma*1000000/(np.pi)
def s_nu(f,sigma):
    return s_f(f,sigma)/(f*f)

#ode solving
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
        derivs=[0.5*1j*omega*np.exp(1j*(delta*t+phigen(t,phase_arr)))*b,
                0.5*1j*omega*np.exp(-1j*(delta*t+phigen(t,phase_arr)))*a]
        return derivs

    def jac(t,y):
        a=y[0]
        b=y[1]
        return [[0,0.5*1j*omega*np.exp(1j*(delta*t+phigen(t,phase_arr)))],
                [0.5*1j*omega*np.exp(-1j*(delta*t+phigen(t,phase_arr))),0]]

    #intitial condition
    y0=[1+0.0j,0+0.0j]
    t0=[0,tpi]

    sol = solve_ivp(feq,t0,y0,rtol=1e-7,atol=3e-7)
    yf=(sol.y[0][-1],sol.y[1][-1])
    #print(yf)
    return [(abs(yf[0]))**2,(abs(yf[1]))**2]

#linewidth sweep    
def swp_lw(lw):
    for k in range(nf):
        s_nu_arr[k] = 2*np.sqrt(s_nu((k+0.000005)*df,lw)*df)
        #s_nu_arr[k] = 2*np.sqrt(s_nu((k+1)*df,lw)*df)
    res=[]
    for count in range(nrun):
        res.append(run_job())
    totresult=np.array(res)
    r0=totresult[:,1]
    meanr0=np.mean(r0)
    stdp=0.001
    stdn=0.001
    return [meanr0,stdp,stdn]


x_fmax=[]
y1_fmax=[]
sp_fmax=[]
sn_fmax=[]
y2_fmax=[] 
y3_fmax=[]
s2_fmax=[]
for i in range(10):
    print(i)
    tpi=2*tpi0
    lwset=0.05*(i+1)
    #delta=2*np.pi*np.sqrt(s_f(0,lwset)*df)
    print(delta)
    res=swp_lw(lwset)
    y1_fmax.append(res[0])
    sp_fmax.append(res[1])
    sn_fmax.append(res[2])
#    lwfactor=1/((i+1)*0.1)
#    res=swp_lw(lwset)
#    y2_fmax.append(res[0])
#    s2_fmax.append(res[1])
    #y3_fmax.append(lwset**2)
    y3_fmax.append(0.75*1*(np.pi)**2*lwset**4)
    x_fmax.append(i+1)
plt.plot(x_fmax,y1_fmax,label="Simulation")

#for i in range(10):
#    plt.plot([x_fmax[i],x_fmax[i]],[y1_fmax[i]-sn_fmax[i],y1_fmax[i]+sp_fmax[i]],'r')
#plt.plot(x_fmax,y2_fmax,label="Quasi static 1")
plt.plot(x_fmax,y3_fmax,label="New quasi static")
plt.yscale('log')
plt.xlabel("σ")
plt.ylabel("error")
plt.title("Error at t=2π/Ω")
plt.legend(loc='upper right')
plt.show()
#For parallel use below
##num_cores = 1
##results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
##results = np.array(results)
##print("Averaged population for each state:")
##print("y[0]="+str(np.mean(results[:,0])) + "+-" + str(np.std(results[:,0])/np.sqrt(nrun)))
##print("y[1]="+str(np.mean(results[:,1]))+ "+-" + str(np.std(results[:,1])/np.sqrt(nrun)))

