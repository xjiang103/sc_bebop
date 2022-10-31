from joblib import Parallel, delayed #for parallel run
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

import argparse
parser = argparse.ArgumentParser(description='Laser Noise')
parser.add_argument('-nrun','--n_run_times', help='Number of Averages', required=True, type=int)
args = vars(parser.parse_args())
b = args["n_run_times"]


num_cores=64
nrun=b
units=1e6
f=open("f3swp2_0810_e.txt","w")
avenum=1

#rabi parameters
omega1=2*np.pi*(44.72)*(units)
omega2=2*np.pi*(44.72)*(units)
delta1=2*np.pi*(1000)*(units)
delta2=delta1*(1-np.sqrt(1+(omega1**2-omega2  **2)/(2*delta1**2)))-delta1
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

tpi0=np.pi/omegar
tpi=tpi0
tpinum=2
print(tpi)

#frequency domain sample parameters
fmax=10*omegar/(2*np.pi)
nf=100000
fmin=fmax/nf
df=fmin

fcmin=0.3*omegar/(2*math.pi)
fcmin=142000
fcmax=6.0*omegar/(2*math.pi)
dfc=0.05*omegar/(2*math.pi)
nl=1
nl=int((fcmax-fcmin)/dfc)
print("nl is "+str(nl))
sigmafc=0.05*omegar/(2*math.pi)

omega_0=omegar

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwset=0.01
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

##def s_nu(f,lw):
##    return lw/(f*f*math.pi)
def frac_power(hg,sigmag,fg):
    return np.sqrt(np.sqrt(2*np.pi))*hg*sigmag/fg**2

def s_nu(f,lw):
   # sigmafc=lw/4
    hg0=1100
    fg0=234*(1e3)
    sigmag=1.4*(1e3)

    scale_fac=lw/fg0

    hg=hg0*(scale_fac**2)
    fg=lw
    

    snu=hg*np.exp(-(f-fg)**2/(2*sigmag**2))
    
    return snu/(f*f)
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

        return derivs

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
    #For parallel use below
    res=[]
    results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
    res = np.array(results)
##    for count in range(nrun):
##        res.append(run_job())

    totresult=np.array(res)
    r0=totresult[:,2]
    meanr0=np.mean(r0)

    sump=0.0
    npo=0.0
    sumn=0.0
    nn=0.0
    for k in range(nrun):
        if (r0[k]>meanr0):
            npo=npo+1
            sump=sump+(r0[k]-meanr0)**2
        else:
            nn=nn+1
            sumn=sumn+(r0[k]-meanr0)**2

    stdp=0
    stdn=0
    
    return [meanr0,(np.std(totresult,axis=0))[2],stdp,stdn]

timet=time.time()
swparr=[[10000,1],[1000,1],[100,1],[10000,2],[1000,2],[100,2]]
x_fmax=[]
x4_fmax=[]
y1_fmax=[]
sp_fmax=[]
sn_fmax=[]
y2_fmax=[]
y3_fmax=[]
y4_fmax=[]
y4_stdp=[]
y4_stdn=[]
s2_fmax=[]
eprev=0
eprevacc=0
stdpacc=0
stdnacc=0

fg_start=234*(1e3)
nl=40
for i in range(nl):
#    tpi=tpi0*swparr[i][1]
#    lwset=swparr[i][0]
    print(i)
    
    tpi=tpinum*tpi0
    lwset=2*(omega_0/(2*np.pi))+6*(omega_0/(2*np.pi))*(i+1)/nl
    print(lwset/(omega_0/(2*np.pi)))
    
    res=swp_lw(lwset,lwset)
    print(res[0])
#    print(res[1])
    y1_fmax.append(res[0])
    sp_fmax.append(res[1])
    sn_fmax.append(res[2])
    x_fmax.append(lwset/(omega_0/(2*math.pi)))


    hg0=1100
    fg0=234*(1e3)
    sigmag=1.4*(1e3)

    scale_fac=lwset/fg0

    hg=hg0*(scale_fac**2)
    fg=lwset
    
    print(frac_power(hg,sigmag,fg))
for i in range(nl):
    f.write(str(x_fmax[i])+' '+str(y1_fmax[i])+'\n')
f.close()
#plt.plot(x_fmax,y1_fmax,lw=2,label=r"simulation")
#plt.plot(x_fmax,y2_fmax,lw=2,label=r"quasi-static")
#plt.plot(x_fmax,y3_fmax,lw=2,label=r"quasi-static1")
##plt.plot(x_fmax,y1_fmax,'o-')
##plt.xlabel("fg/omega_R")
##plt.ylabel("Error")
###plt.yscale("log")
###plt.xscale("log")
##plt.title("Error at time=2π/Ω, sweeping fg")
##plt.show()

