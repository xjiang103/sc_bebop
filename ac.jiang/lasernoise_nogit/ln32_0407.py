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
parser = argparse.ArgumentParser(description='Length of pulse/(π/Ω)')
parser.add_argument('-tpi','--tpi_number', help='Tpi_number', required=True, type=int)
args = vars(parser.parse_args())
tpi_num = args["tpi_number"]

num_cores=8
nrun=20
units=1e6
filestr="f3swp"+str(tpi_num)+"_0407.txt"
f=open(filestr,"w")
print(filestr)
avenum=1

scale_fac0=1.0

hg=1100
fg=234*(1e3)
sigmag=1.4*(1e3)
#rabi parameters

delta1=2*np.pi*(500000)*(units)
omega1=np.sqrt(2*500000)*2*np.pi*units
delta1=0.5*10000*2*np.pi*units
omega1=100*2*np.pi*units
omega2=omega1
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
print(delta/omega1)

tpi0=np.pi/omegar
tpi=tpi0
tpinum=tpi_num
print(tpi)

#frequency domain sample parameters
fmax=10*omegar/(2*np.pi)
nf=100000
fmin=fmax/nf
df=fmin


scale_fac0=1.0

hg0=200


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

def s_nu(f,fg):
    snu=hg
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
    if ((tpinum %2)==1):
        r0=totresult[:,0]
    else:
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
    stdp=np.sqrt(sump/float(npo))
    stdn=np.sqrt(sumn/float(nn))
    return [meanr0,(np.std(totresult,axis=0))[2],stdp,stdn]

timet=time.time()
swparr=[[10000,1],[1000,1],[100,1],[10000,2],[1000,2],[100,2]]

x_f=[]
y_f=[]
sn=[]
sp=[]


fg_start=234*(1e3)
nl=20
for i in range(20):
    print(i)
    tpi=tpi0*tpinum
    #
    #scale_fac=scale_fac_list[i]
    hg=1*hg0*((i+1))
    scale_fac=1
    res=swp_lw(scale_fac*fg,scale_fac*fg)
    
    y_f.append(res[0])
    x_f.append(0)
    sp.append(res[1])
    sn.append(res[2])
    f.write(str(x_f[i])+" "+str(y_f[i])+" "+str(sp[i])+" "+str(sn[i])+"\n")
    print("-------")
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

