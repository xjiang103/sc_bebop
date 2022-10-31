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

#number of runs for averaging
num_cores=8
nrun=100

avenum=1
f=open("fampswp_0303.txt","w")
#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=math.pi/(omega_0)
tpi=tpi0
#frequency domain sample parameters
fmax=0.001*omega_0/(2*math.pi)

nf=1000
fmin=fmax/nf
df=fmin

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwset=0.3*omega_0/(2*math.pi)
lwfactor=1
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
def s_nu(f,lw):
    return lw
   # sigmafc=lw/4
    #snu=100+(f*f/(1))*1*(1)*(1/(np.sqrt(2*np.pi)*sigmafc))*np.exp(-(f-fg)**2/(2*sigmafc**2))/(2*np.sqrt(2*np.pi*sigmafc*sigmafc))
    snu=0
    ha=500
    hb=150
    fj=300
    fe=40000
    d1=10000
    hg=2500
    fg=lw
    sigmag=12*1000
    hg1=7.52*(1e7)
    
    #snu=0*ha*(1+fj/f)+0*(1/2)*(hb-ha)*(erf((f-fe)/d1)-1)+(hg)*np.exp(-(f-fg)**2/(2*sigmag**2))

    #snu=ha*(1+fj/f)+(1/2)*(hb-ha)*(erf((f-fe)/d1)-1)+(hg1*f**2/(fg**2*sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)

    snu=(hg1*f**2/(fcmin**2*sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)
    
    return snu/(f*f)

omegascale=1
#ode solving
def run_job():
    #print("entering run_job")
    #print("run #"+str(count)+"/"+str(nrun))

    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df.
    phase_arr=np.random.uniform(0,2*np.pi,size=nf)
    #print("working in run_job")
    
    def feq(t,y):
        a=y[0]
        b=y[1]
        omega=omega_0*np.sqrt(1+phigen(t,phase_arr))
        derivs=[0.5*1j*omega*np.exp(1j*1)*b,
                0.5*1j*omega*np.exp(-1j*1)*a]
        return derivs

    def jac(t,y):
        a=y[0]
        b=y[1]
        omega=omega_0*np.sqrt(1+phigen(t,phase_arr))
        return [[0,0.5*1j*omega*np.exp(1j*1)],
                [0.5*1j*omega*np.exp(-1j*1),0]]

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
    
    r0=totresult[:,1]
    meanr0=np.mean(r0)
    sump=0.0
    npo=0.0
    sumn=0.0
    nn=0.0
    stdp=0
    stdn=0
##    for k in range(nrun):
##        if (r0[k]>meanr0):
##            npo=npo+1
##            sump=sump+(r0[k]-meanr0)**2
##        else:
##            nn=nn+1
##            sumn=sumn+(r0[k]-meanr0)**2
##    stdp=np.sqrt(sump/float(npo))
##    stdn=np.sqrt(sumn/float(nn))
    return [meanr0,stdp,stdn]


x_fmax=[]
y1_fmax=[]
y2_fmax=[]

nl=20
for i in range(nl):
#    tpi=tpi0*swparr[i][1]
#    lwset=swparr[i][0]
    rin=0.1*(i+1)/nl
    
    tpi=2*tpi0
    
    lwset=rin*rin/fmax

    for k in range(nf):
        pikdf_arr[k] = 2*np.pi*(k+1)*df
#    print(")*(&*")

    res=swp_lw(lwset)
    print(res[0])
#    print(res[1])
    y1_fmax.append(res[0])
    
    sig=np.sqrt(lwset*fmax)
    y2_fmax.append(0.25*(((2/2)*np.pi)**2)*sig**2)
#    y2_fmax.append(sig**2+3*((6.5*np.pi)**2/4-1)*sig**4)
#    y3_fmax.append(quad(integrand,0,np.inf,args=(6.5,lwset*nf*df/(omega_0/(2*np.pi))))[0])
    print("-----")
    x_fmax.append(rin)


plt.plot(x_fmax,y1_fmax,label="Simulation") 
plt.plot(x_fmax,y2_fmax,label="quasi-static") 
##for i in range(int(nl/avenum)):
##    f.write(str(x_fmax[i])+" "+str(y1_fmax[i])+" "+str(y4_stdn[i])+" "+str(y4_stdp[i])+"\n")
##    #plt.plot([x4_fmax[i],x4_fmax[i]],[y4_fmax[i]-y4_stdn[i],y4_fmax[i]+y4_stdp[i]],'r')
##f.close()
plt.legend()
plt.yscale('log')
plt.xlabel("RIN level")
plt.ylabel('error')
plt.title("Intensity noise, 2π pulse")
plt.show()
#For parallel use below
##num_cores = 1
##results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
##results = np.array(results)
##print("Averaged population for each state:")
##print("y[0]="+str(np.mean(results[:,0])) + "+-" + str(np.std(results[:,0])/np.sqrt(nrun)))
##print("y[1]="+str(np.mean(results[:,1]))+ "+-" + str(np.std(results[:,1])/np.sqrt(nrun)))

