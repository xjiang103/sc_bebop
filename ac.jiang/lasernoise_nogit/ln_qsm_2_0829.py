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
import argparse

parser = argparse.ArgumentParser(description='Length of pulse/(π/Ω)')
parser.add_argument('-tpi','--tpi_number', help='Tpi_number', required=True, type=int)
args = vars(parser.parse_args())
tpi_num = args["tpi_number"]

nrun=200
num_cores=64

tpnum=tpi_num
filestr="fmaxswp_0830_"+str(tpnum)+".txt"
f=open(filestr,"w")
units=1e6

#rabi parameters
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
omega_0=omegar

tpi0=np.pi/omegar
tpi=tpi0
print(tpi)

#frequency domain sample parameters
fmax=0.01*omega_0/(2*math.pi)

nl=25
nf=10000
fmin=fmax/nf
df=fmin

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwfactor=1
#pre-calculating some constants

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
def s_nu(f,lw):
    return lw*lwfactor/(f*f*math.pi)

#ode solving
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


#linewidth sweep    
def swp_lw(lw1,lw2):
    for k in range(nf):
        s_nu_arr1[k] = 2*np.sqrt(s_nu((k+1)*df,lw1)*df)
        s_nu_arr2[k] = 2*np.sqrt(s_nu((k+1)*df,lw2)*df)
    res=[]

##    for count in range(nrun):
##        res.append(run_job())
    
    results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
    res = np.array(results)
    
    totresult=np.array(res)
    r0=[]
    if (tpnum%2==1):
        r0=totresult[:,0]
    elif(tpnum%2==0):
        r0=totresult[:,2]
    meanr0=np.mean(r0)

    stdp=0
    stdn=0
    return [meanr0,stdp,stdn]

timet=time.time()
swparr=[[10000,1],[1000,1],[100,1],[10000,2],[1000,2],[100,2]]
x1=[]
y1=[]
sp=[]
sn=[]
y2=[]
sigarr=[]
quasarr=[]


for i in range(nl):
    print(i)
    tpi=tpnum*tpi0
    lwset1=100000*(i+1)
    lwset2=100000*(i+1)
    lwfactor=1
    res=swp_lw(lwset1,lwset2)
    print(res[0])
    y1.append(res[0])
    sp.append(res[1])
    sn.append(res[2])
    sig1=np.sqrt((1/np.pi)*2*lwset1*nf*df)/(omega_0/(2*np.pi))
    sig2=np.sqrt((1/np.pi)*2*lwset1*nf*df)/(omega_0/(2*np.pi))
    sig=np.sqrt(sig1**2+sig2**2)
    quas=0
    if (tpnum%2==1):
        quas=sig**2
    elif(tpnum%2==0):
        quas=0.75*(tpnum*np.pi/2)**2*sig**4
    print(quas)
    quasarr.append(quas)
    sigarr.append(sig)
    x1.append(lwset1)


for i in range(nl):
    f.write(str(x1[i])+' '+str(y1[i])+' '+str(quasarr[i])+'\n')
f.close()
