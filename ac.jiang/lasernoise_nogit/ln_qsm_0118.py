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

tpnum=tpi_num

nrun=300
num_cores=64

#tpnum=1
filestr="fmaxswp_0118_"+str(tpnum)+".txt"
f=open(filestr,"w")

#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=math.pi/(omega_0)
tpi=tpi0
#frequency domain sample parameters
fmax=0.01*omega_0/(2*math.pi)

nl=80
nf=10000
fmin=fmax/nf
df=fmin
df0=df

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
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
    return lw*lwfactor/(f*f*math.pi)

#ode solving
def run_job():
    #print("run #"+str(count)+"/"+str(nrun))

    #for each run, pre-calculate the extra random phase for each frequency component
    #f_k=(k+1)*df.
    phase_arr=np.random.uniform(0,2*np.pi,size=nf)

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
    r0=[]
    if (tpnum%2==1):
        r0=totresult[:,0]
    elif(tpnum%2==0):
        r0=totresult[:,1]
    meanr0=np.mean(r0)
    #print(meanr0)
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
    lwset=10000*(5)
    lwfactor=1
    df=df0*(i+1)
    fmax_tmp=df*nf
    for k in range(nf):
        pikdf_arr[k] = 2*np.pi*(k+1)*df
    res=swp_lw(lwset)
    print(res[0])
    y1.append(res[0])
    sp.append(res[1])
    sn.append(res[2])
    sig=np.sqrt((1/np.pi)*2*lwset*nf*df)/(omega_0/(2*np.pi))
    quas=0
    if (tpnum%2==1):
        quas=sig**2
    elif(tpnum%2==0):
        quas=0.75*(tpnum*np.pi/2)**2*sig**4
    print(quas)
    quasarr.append(quas)
    sigarr.append(sig)
    x1.append(fmax_tmp)


for i in range(nl):
    f.write(str(x1[i])+' '+str(y1[i])+' '+str(quasarr[i])+' '+str(sp[i])+' '+str(sn[i])+'\n')
f.close()
