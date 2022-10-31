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

nrun=100
nphase_ave=10
num_cores=64
f=open("ln2c1_0108.txt","w")


omega0=(1e6)*2*np.pi
tpi=np.pi/np.abs(omega0)
omega=omega0

#gap time sweep
Tmax=8*tpi
nTsweep=10
ngdT=Tmax/nTsweep
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
s_nu_arr= np.zeros(nf)
for k in range(nf):
    pikdf_arr[k] = 2*np.pi*(k+1)*df

np.random.seed()
#function that returns the time-varying phasenoise
def phigen(t,phase_arr):
    phitemp = np.sum(s_nu_arr*np.cos(pikdf_arr*t+phase_arr))
    return phitemp

##def s_nu(f,lw):
##     return lw/(f*f*math.pi)
def s_nu(f,lw):
    snu=lw/np.pi
    
    return snu/(f*f)

def func(x,a,b,c,d):
    return d+c*np.cos(a*x+b)

def func2(x,T2):
    return np.exp(-x/T2)

def run_job(T,Rphase):
    #print("run #"+str(count)+"/"+str(nrun)) 
    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df.
    phase_arr=np.random.uniform(0,2*np.pi,size=nf)
    omega=omega0
    def feq(t,y):
        a=y[0]
        b=y[1]
        phi =phigen(t,phase_arr)
        derivs=[0.5*1j*np.conj(omega)*np.exp(1j*(phi))*b,
                0.5*1j*omega*np.exp(-1j*(phi))*a]
        return derivs
    #intitial condition
    #1st pi/2 pulse
    y0=[1+0.0j,0+0.0j]
    t0f = [0,tpi/2]

    sol1 = solve_ivp(feq,t0f,y0,rtol=1e-7,atol=3e-7)
    y0f = [sol1.y[0][-1],sol1.y[1][-1]]
    #2nd pi/2 pulse
    omega=omega0*np.exp(Rphase*1j)
    y1=y0f
    t1f=[tpi/2+T,tpi+T]
    sol2 = solve_ivp(feq,t1f,y1,rtol=1e-7,atol=3e-7)
    y1f = [sol2.y[0][-1],sol2.y[1][-1]]
    
    return [(abs(y1f[0]))**2,(abs(y1f[1]))**2]

lw_arr=np.linspace(dlw,lwmax,nlwsweep)
T2_arr=[]
:r k in range(nf):
        s_nu_arr[k] = 2*np.sqrt(s_nu((k+1)*df,lw)*df)
    print("Linewidth="+str(lw)+"Hz")
    x_T=np.linspace(dT,Tmax,nTsweep)
    np_T=[]

    #sweep gap time T: get the fringe amplitude at different T to
    #extract T2 by fitting to Amplitude=exp(-T/T2)
    for i_T in range(nTsweep):
        T=(i_T+1)*dT
        ave_phase=[]

        #Do multiple phase sweep and take the average
        for i_pave in range(nphase_ave):
            x_phase=np.linspace(dRphase,Rphasemax,nRphasesweep)
            p_phase=[]

            #sweep the phase to get the Ramsey fringe(Population-Rphase)
            #Then do the fit to cos(Rphase)to exctract fringe amplitude
            for i_Rphase in range(nRphasesweep):
                Rphase=(i_Rphase+1)*dRphase
                res=[]

                #do multiple runs and get the aveage population for the
                #upper level
##                for i_nrun in range(nrun):
##                    res.append(run_job(T,Rphase))
##                results=np.array(res)
                results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)(T,Rphase) for count in range(nrun))
                results = np.array(results)
                
                r0=results[:,1]
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

