from joblib import Parallel, delayed #for parallel run
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy import stats
import argparse
plt.rcParams.update({'font.size': 22})


parser = argparse.ArgumentParser(description='Length of pulse/(π/Ω)')
parser.add_argument('-tpi','--tpi_number', help='Tpi_number', required=True, type=int)
args = vars(parser.parse_args())
tpi_num = args["tpi_number"]


#number of runs for averaging
nrun=100
num_cores=64
tpnum=tpi_num
filestr="h0swp_0224_"+str(tpnum)+".txt"
f=open(filestr,"w")

#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=1*math.pi/(omega_0)
tpi=tpi0
#frequency domain sample parameters
fmax=10*omega_0/(2*math.pi)
nf=1000000
fmin=fmax/nf
df=fmin

#noise spectrum parameters
scale_fac0=1.0

hg0=200
fg=1000*(1e3)*1.25
sigmag=1.5*(1e3)

#pre-calculating some constants

pikdf_arr= np.zeros(nf)
s_nu_arr= np.zeros(nf) 
for k in range(nf):
    pikdf_arr[k] = 2*np.pi*(k+1)*df

np.random.seed()
#function that returns the time-varying phasenoise
def frac_power(hg,sigmag,fg):
    return np.sqrt(np.sqrt(2*np.pi))*hg*sigmag/fg**2
def phigen(t,phase_arr):
    phitemp = np.sum(s_nu_arr*np.cos(pikdf_arr*t+phase_arr))
    return phitemp
def s_nu(f,fg):
    snu=hg
    return snu/(f*f)
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
    
    results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
    res = np.array(results)
    
##    for count in range(nrun):
##        res.append(run_job())

        
    totresult=np.array(res)
    if ((tpnum %2)==1):
        r0=totresult[:,0]
    else:
        r0=totresult[:,1]
    
    meanr0=np.mean(r0)
    print(meanr0)
    sump=0.0
    npo=1.0
    sumn=0.0
    nn=1.0
    stdp=0
    stdn=0
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

x_f=[]
y_f=[]
stdp_f=[]
stdn_f=[]

nl=20
for i in range(20):
    print(i)
    tpi=tpi0*tpnum
    
    #scale_fac=scale_fac_list[i]
    
    hg=1*hg0*((i+1))
    res=swp_lw(fg)
    print(frac_power(hg,sigmag,fg))
    print(1-res[0])
    
    y_f.append(res[0])
    x_f.append(frac_power(hg,sigmag,fg))
    stdp_f.append(res[1])
    stdn_f.append(res[2])
    print("-------")

for i in range(nl):
    f.write(str(x_f[i])+' '+str(y_f[i])+' '+str(stdp_f[i])+' '+str(stdn_f[i])+'\n')
f.close()
plt.plot(x_f,y_f,'o-')
##for i in range(int(nl)):
##    plt.plot([x_f[i],x_f[i]],[y_f[i]+stdp_f[i],y_f[i]-stdn_f[i]],'r')
plt.xlabel("Frac_Power")
plt.ylabel("Error")
plt.yscale("log")
plt.xscale("log")
plt.title("Error at time="+str(tpnum)+"π/Ω, sweeping hg")
plt.show()
