from joblib import Parallel, delayed #for parallel run
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy import stats

f=open("f3qs_0218.txt","w")

units=1e0

num_cores=8
nrun=10
timenum=2
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

alpha=(omega1**2-omega2**2)/(2*delta**2)
print("alpha="+str(alpha))

print(omega2)

tpi0=np.pi/omegar
tpi=tpi0
print(tpi)

#frequency domain sample parameters
fmax=0.1*omegar/(2*np.pi)
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
#    for count in range(nrun):
#        res.append(run_job())
    results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
    res = np.array(results)

    totresult=np.array(res)
    r0=totresult[:,0]
    meanr0=np.mean(r0)

    #stdp=np.sqrt(sump/float(npo))
    #stdn=np.sqrt(sumn/float(nn))
    stdp=0
    stdn=0
    
    return [meanr0,(np.std(totresult,axis=0))[2],stdp,stdn]

timet=time.time()
nl=10
xarr=[]
y1arr=[]
yqsarr=[]
for i in range(nl):
    print(i)
    tpi=timenum*tpi0

    lwset=0.0001*(2**(i))
    print(lwset)
    results = swp_lw(lwset,lwset)

    mean_arr=1-results[0]
    print("error=  "+str(mean_arr))
    std_arr=results[1]
    std_arrp=results[2]
    std_arrn=results[3]
    xarr.append(lwset*1000)
    y1arr.append(mean_arr)

    sig1=np.sqrt((lwset*fmax/np.pi)/(omegar/(2*np.pi*units))**1)
    sig2=np.sqrt((lwset*fmax/np.pi)/(omegar/(2*np.pi*units))**1)

    sig=np.sqrt((1-alpha)**2*sig1**2+(1+alpha)**2*sig2**2)
    yqsarr.append(0.75*0.25*timenum**2*(np.pi)**2*sig**4)


for i in range(int(nl)):
    f.write(str(xarr[i])+" "+str(y1arr[i])+" "+str(yqsarr[i])+"\n")
    
plt.plot(xarr,y1arr,lw=2,label="Simulation")
plt.plot(xarr,yqsarr,lw=2,label="Quasi-Static")
plt.yscale('log')
plt.legend()
plt.show()

