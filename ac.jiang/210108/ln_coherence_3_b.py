import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy import stats

from joblib import Parallel, delayed #for parallel run

nrun=5
nphase_ave=2
num_cores=64
f=open("ln3c1_0108.txt","w")

units=1e6

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

omega0=omegar
#gap time sweep
Tmax=8*tpi
nTsweep=10
dT=Tmax/nTsweep
#frequency domain sampling
fmax=10*omega0/(2*np.pi)
nf=1000
fmin=fmax/nf
df=fmin
#parameter for phase sweep
Rphasemax=4*np.pi
nRphasesweep=10
dRphase=Rphasemax/nRphasesweep
#lindwidth of laser
lwmax=0.1*omega0/(2*np.pi)
nlwsweep=8
dlw=lwmax/nlwsweep


pikdf_arr= np.zeros(nf)
s_nu_arr1= np.zeros(nf)
s_nu_arr2= np.zeros(nf)
for k in range(nf):
    pikdf_arr[k] = 2*np.pi*(k+1)*df

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
def s_nu(f,lw):
   # sigmafc=lw/4
    #snu=100+(f*f/(1))*1*(1)*(1/(np.sqrt(2*np.pi)*sigmafc))*np.exp(-(f-fg)**2/(2*sigmafc**2))/(2*np.sqrt(2*np.pi*sigmafc*sigmafc))
    return lw/(f*f*math.pi)
    snu=0
    ha=500
    hb=150
    fj=300
    fe=40000
    d1=10000
    hg=2500
    fg=lw
    #sigmag=12*1000
    sigmag=fg*0.084507

    hg1=7.52*(1e7)
    
    #snu=1*ha*(1+fj/f)+1*(1/2)*(hb-ha)*(erf((f-fe)/d1)-1)+(hg)*np.exp(-(f-fg)**2/(2*sigmag**2))

    #snu=(hg1/(sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)

    fg=142000

    hg1=7.52*(1e9)*lw/fg

    snu=(hg1/(sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)

    #snu=(hg1*f**2/(fcmin**2*sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)
    
    return snu/(f*f)

def func(x,a,b,c,d):
    return d+c*np.cos(a*x+b)

def func2(x,T2):
    return np.exp(-x/T2)

def run_job(T,Rphase):
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
        
    omega=omega0

    #1st pi/2 pulse
    y0=[1+0.0j,0+0.0j,0+0.0j]
    t0f = [0,tpi/2]

    sol1 = solve_ivp(feq,t0f,y0,rtol=1e-7,atol=3e-7)
    y0f = [sol1.y[0][-1],sol1.y[1][-1],sol1.y[2][-1]]
    #2nd pi/2 pulse
    omega=omega0*np.exp(Rphase*1j)
    y1=y0f
    t1f=[tpi/2+T,tpi+T]
    sol2 = solve_ivp(feq,t1f,y1,rtol=1e-7,atol=3e-7)
    y1f = [sol.y[0][-1],sol.y[1][-1],sol.y[2][-1]]
    
    return [(abs(yf[0]))**2,(abs(yf[1]))**2,(abs(yf[2]))**2]

lw_arr=np.linspace(dlw,lwmax,nlwsweep)
T2_arr=[]

#sweep lindwidth: look at T2 at different lw(noise level)
for i_lw in range(nlwsweep):
    lw=(i_lw+1)*dlw

    for k in range(nf):
        s_nu_arr1[k] = 2*np.sqrt(s_nu((k+1)*df,lw)*df)
        s_nu_arr2[k] = 2*np.sqrt(s_nu((k+1)*df,lw)*df)
    print("Linewidth="+str(lw)+"Hz")
    x_T=np.linspace(dT,Tmax,nTsweep)
    p_T=[]

    #sweep gap time T: get the fringe amplitude at different T to
    #extract T2 by fitting to Amplitude=exp(-T/T2)
    for i_T in range(nTsweep):
        print("i_T="+str(i_T))
        T=(i_T+1)*dT
        ave_phase=[]

        #Do multiple phase sweep and take the average
        for i_pave in range(nphase_ave):
            print("i_pave="+str(i_pave))
            x_phase=np.linspace(dRphase,Rphasemax,nRphasesweep)
            p_phase=[]

            #sweep the phase to get the Ramsey fringe(Population-Rphase)
            #Then do the fit to cos(Rphase)to exctract fringe amplitude
            for i_Rphase in range(nRphasesweep):
                print("i_Rphase="+str(i_Rphase))
                Rphase=(i_Rphase+1)*dRphase
                res=[]

                #do multiple runs and get the aveage population for the
                #upper level
                for i_nrun in range(nrun):
                    res.append(run_job(T,Rphase))
                results=np.array(res)
##                results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)(T,Rphase) for count in range(nrun))
##                results = np.array(results)
                
                r0=results[:,2]
                p_phase.append(np.mean(r0))

            #fit to c*cos(a*Rphase+b)+d 
            popt,pcov=curve_fit(func,x_phase,p_phase)
            ave_phase.append(np.abs(2*popt[2]))
        p_T.append(np.mean(ave_phase))
        print("T= "+str(T/tpi)+", amp= "+str(p_T[i_T]))
    popt2,pcov2=curve_fit(func2,x_T,p_T)
    T2_arr.append(popt2[0])
    f.write(str(lw_arr[i_lw])+" "+str(T2_arr[i_lw])+"\n")

f.close()

