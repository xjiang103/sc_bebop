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
omega1=2*np.pi*(44.72)*(units)
omega2=2*np.pi*(44.72)*(units)
delta1=2*np.pi*(1000)*(units)
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

tpi0=np.pi/omegar
tpi=tpi0
print(tpi)

#frequency domain sample parameters
fmax=10*omegar/(2*np.pi)
nf=1000
fmin=fmax/nf
df=fmin
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

def s_nu(f,lw):
    return lw/(f*f*math.pi)

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
    for count in range(nrun):
        res.append(run_job())

    totresult=np.array(res)
    r0=totresult[:,0]
    meanr0=np.mean(r0)
    meanr1=np.mean(totresult[:,1])
    meanr2=np.mean(totresult[:,2])
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
    
    return [meanr0,(np.std(totresult,axis=0))[2],stdp,stdn,meanr1,meanr2]

timet=time.time()
swparr=[[0.01,1],[0.001,1],[0.0001,1],[0.01,2],[0.001,2],[0.0001,2]]
for i in range(6):
    print(i)
    tpi=tpi0*swparr[i][1]
    lwset=0
    results = swp_lw(lwset,lwset)
    mean_arr=results[0]
    std_arr=results[1]
    std_arrp=results[2]
    std_arrn=results[3]
    m1=results[4]
    m2=results[5]
    print("------")
    if(swparr[i][1]==1):
        print("pi pulse")
    else:
        print("2pi pulse")
    print("FWHM=",lwset*1000,"kHz")
    print("P[0]:",mean_arr)
    print("P[r]:",m2)
    print("P[int]:",m1)
    print("Standard Deviation:",std_arr)
    if(swparr[i][1]==1):
        print("Standard Deviation P:",std_arrp)
        print("Standard Deviation N:",std_arrn)
    else:
        print("Standard Deviation P:",std_arrn)
        print("Standard Deviation N:",std_arrp)        

