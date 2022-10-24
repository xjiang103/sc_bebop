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
num_cores=64
nrun=800

avenum=1
f=open("fswp3_0127.txt","w")
#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=math.pi/(omega_0)
tpi=tpi0
#frequency domain sample parameters
fmax=10*omega_0/(2*math.pi)

fcmin=0
fcmax=0.001*omega_0/(2*math.pi)
dfc=0.00005*omega_0/(2*math.pi)
nl=1
nl=int((fcmax-fcmin)/dfc)
print("nl is "+str(nl))
sigmafc=0.05*omega_0/(2*math.pi)

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
##def s_nu(f,lw):
##   # sigmafc=lw/4
##    #snu=100+(f*f/(1))*1*(1)*(1/(np.sqrt(2*np.pi)*sigmafc))*np.exp(-(f-fg)**2/(2*sigmafc**2))/(2*np.sqrt(2*np.pi*sigmafc*sigmafc))
##    snu=0
##    ha=500
##    hb=150
##    fj=300
##    fe=40000
##    d1=10000
##    hg=2500
##    fg=lw
##    #sigmag=12*1000
##    sigmag=fg*0.084507
##
##    hg1=7.52*(1e7)
##    
##    #snu=1*ha*(1+fj/f)+1*(1/2)*(hb-ha)*(erf((f-fe)/d1)-1)+(hg)*np.exp(-(f-fg)**2/(2*sigmag**2))
##
##    snu=0*ha*(1+fj/f)+0*(1/2)*(hb-ha)*(erf((f-fe)/d1)-1)+(hg1/(sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)
##
##    #snu=(hg1*f**2/(fcmin**2*sigmag*np.sqrt(2*np.pi)))*np.exp(-(f-fg)**2/(2*sigmag)**2)
##
##    if(f<fmax/1000):
##        return 0
##    else:
##        return snu/(f*f)
def s_nu(f,lw):
    snu=0
    if ((f>lw) and (f<lw+dfc)):
        snu=1*(1e4)
    else:
        snu=0
    return snu/(f*f)



#ode solving
def run_job():
    #print("entering run_job")
    #print("run #"+str(count)+"/"+str(nrun))

    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df.
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
    meanr0=np.mean(r0)
    stdp=0.1
    stdn=0.1
##    sump=0.0
##    npo=0.0
##    sumn=0.0
##    nn=0.0
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

def integrand(x,n,s):
    return (1/np.sqrt(2*np.pi*s**2))*np.exp(-x**2/(2*s**2))*(1-(1/1+x**2)*(np.cos(omega_0*np.sqrt(1+x**2)*n*np.pi))**2)
for i in range(nl):
#    tpi=tpi0*swparr[i][1]
#    lwset=swparr[i][0]
    print(i)
    
    tpi=2*tpi0
    lwset=fcmin+(i)*dfc
    print(lwset)
    for k in range(nf):
        pikdf_arr[k] = 2*np.pi*(k+1)*df
#    print(")*(&*")
    
    res=swp_lw(lwset)
    print(res[0])
#    print(res[1])
    y1_fmax.append(res[0])
    eprevacc=eprevacc+res[0]
    stdpacc=stdpacc+res[1]
    stdnacc=stdnacc+res[2]
    if ((i+1)%avenum==0):
        y4_fmax.append(eprevacc/avenum)
        y4_stdp.append(stdpacc/avenum)
        y4_stdn.append(stdnacc/avenum)
        x4_fmax.append(lwset/(omega_0/(2*math.pi)))
        eprevacc=0
        stdpacc=0
        stdnacc=0
    eprev=res[0]
    sp_fmax.append(res[1])
    sn_fmax.append(res[2])
#    lwfactor=1/((i+1)*0.1)
#    res=swp_lw(lwset)
#    y2_fmax.append(res[0])
#    s2_fmax.append(res[1])
    sig=np.sqrt((1/np.pi)*lwset*nf*df)/(omega_0/(2*np.pi))
    y2_fmax.append(0.25*3*((10*np.pi)**2)*sig**4)
#    y2_fmax.append(sig**2+3*((6.5*np.pi)**2/4-1)*sig**4)
#    y3_fmax.append(quad(integrand,0,np.inf,args=(6.5,lwset*nf*df/(omega_0/(2*np.pi))))[0])
    x_fmax.append(lwset/(omega_0/(2*math.pi)))

def func(x,a):
    return a*np.sqrt(x)
popt,pcov=curve_fit(func,x_fmax,y1_fmax)
aaa=popt[0]
print(aaa)
for i in range(nl):
    y3_fmax.append(aaa*np.sqrt(x_fmax[i]))



#plt.plot(x_fmax,y1_fmax,lw=2,label=r"simulation")
#plt.plot(x_fmax,y2_fmax,lw=2,label=r"quasi-static")
#plt.plot(x_fmax,y3_fmax,lw=2,label=r"quasi-static1")
plt.plot(x4_fmax,y4_fmax) 

for i in range(int(nl/avenum)):
    f.write(str(x4_fmax[i])+" "+str(y4_fmax[i])+" "+str(y4_stdn[i])+" "+str(y4_stdp[i])+"\n")
    #plt.plot([x4_fmax[i],x4_fmax[i]],[y4_fmax[i]-y4_stdn[i],y4_fmax[i]+y4_stdp[i]],'r')
f.close()
plt.xlabel("fcenter/(Ω0/2π))")
plt.ylabel('error')
plt.show()
#For parallel use below
##num_cores = 1
##results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
##results = np.array(results)
##print("Averaged population for each state:")
##print("y[0]="+str(np.mean(results[:,0])) + "+-" + str(np.std(results[:,0])/np.sqrt(nrun)))
##print("y[1]="+str(np.mean(results[:,1]))+ "+-" + str(np.std(results[:,1])/np.sqrt(nrun)))

