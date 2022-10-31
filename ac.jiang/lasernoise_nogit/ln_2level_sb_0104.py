from joblib import Parallel, delayed #for parallel run
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp,quad
from scipy import stats
from scipy.special import erf


num_cores=64
#number of runs for averaging
nrun=1000

#number of points for fc sweeping
nl=20

f=open("fswp1.txt","w")

#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=math.pi/(omega_0)
tpi=tpi0

#frequency domain sample parameters
fmax=10*omega_0/(2*math.pi)

nf=10000
fmin=fmax/nf
df=fmin

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
#lwset=0.3*omega_0/(2*math.pi)
#lwfactor=1
#pre-calculating some constants

pikdf_arr= np.zeros(nf)
s_nu_arr= np.zeros(nf) 
for k in range(nf):
    pikdf_arr[k] = 2*np.pi*(k+1)*df

np.random.seed()
#function that returns the time-varying phasenoise
def phigen(t,phase_arr):
    phitemp = np.sum(s_nu_arr*np.cos(pikdf_arr*t+phase_arr))
    return phitemp
def s_nu(f,fc):
    
    ha=500
    hb=150
    fj=300
    fe=40000
    d1=10000
    hg=2500
    fg=fc

    sigmag=fg*0.084507

    hg1=7.52*(1e7)
    
    #snu=1*ha*(1+fj/f)+1*(1/2)*(hb-ha)*(erf((f-fe)/d1)-1)+(hg)*np.exp(-(f-fg)**2/(2*sigmag**2))

    snu=0*ha*(1+fj/f)+0*(1/2)*(hb-ha)*(erf((f-fe)/d1)-1)+(hg1/(sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)
    
    return snu/(f*f)

#ode solving
def run_job():

    phase_arr=np.random.uniform(0,2*np.pi,size=nf)
    #print("working in run_job")
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
        return [[0,0.5*1j*omega*np.exp(1j*phigen(t,phase_arr))],
                [0.5*1j*omega*np.exp(-1j*phigen(t,phase_arr)),0]]

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
        s_nu_arr[k] = 2*np.sqrt(s_nu((k+1)*df,lw)*df)
    res=[]
##    for count in range(nrun):
##        res.append(run_job())
    results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
    res = np.array(results)
    
    totresult=np.array(res)
    
    r0=totresult[:,0]

    #meanr0:population of the ground state
    meanr0=np.mean(r0)

    #calculating the error bars
    #--------
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
    #--------
    
    return [meanr0,stdp,stdn]


x_fmax=[]
y1_fmax=[]
sp_fmax=[]
sn_fmax=[]


#nl: steps for sweeping the bump center frequency
for i in range(nl):

    tpi=1*tpi0

    #sweeping the center frequency, from 360kHz to 5MHz, could be changed
    fcset=0.5*i*fmax/nl+3*120000

    
    for k in range(nf):
        pikdf_arr[k] = 2*np.pi*(k+1)*df

    #run simulation   
    res=swp_lw(fcset)

    
    y1_fmax.append(res[0])
    x_fmax.append(lwset/(omega_0/(2*math.pi)))
    sp_fmax.append(res[1])
    sn_fmax.append(res[2])

plt.plot(x_fmax,y1_fmax) 

for i in range(int(nl)):
    f.write(str(x_fmax[i])+" "+str(y_fmax[i])+" "+str(sn_fmax[i])+" "+str(sp_fmax[i])+"\n")
    #plt.plot([x4_fmax[i],x4_fmax[i]],[y4_fmax[i]-y4_stdn[i],y4_fmax[i]+y4_stdp[i]],'r')
f.close()
plt.xlabel("fcenter/(Ω0/2π))")
plt.ylabel('error')
plt.show()


