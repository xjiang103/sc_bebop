
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy import stats


#number of runs for averaging
nrun=1000

#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=1*math.pi/(omega_0)
tpi=tpi0
#frequency domain sample parameters
fmax=10*omega_0/(2*math.pi)
nf=10000
fmin=fmax/nf
df=fmin

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwset=100
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
    return lw/(f*f*math.pi)
#    return (1*500*(1+1*300/f)-350+1*0.5*(150-500)*(erf((f-40000)/10000)-1)+1*2500*np.exp(-(f-lw)**2/(2*12000**2)))/f**2

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
bucarr=[0]*300
def buc(mean):
    nn=int(-1*math.log10(mean)/(6/300))
    if (nn>299):
        nn=299
    bucarr[299-nn]=bucarr[299-nn]+1
def swp_lw(lw):
    for k in range(nf):
        s_nu_arr[k] = 2*np.sqrt(s_nu((k+1)*df,lw)*df)
    res=[]
    for count in range(nrun):
        res.append(run_job())
    totresult=np.array(res)
    r0=totresult[:,0]
    meanr0=np.mean(r0)
    print(meanr0)
    sump=0.0
    npo=0.0
    sumn=0.0
    nn=0.0
    lerrorcount=0

    for k in range(nrun):
        buc(r0[k])
        if (r0[k]>0.5):
            lerrorcount+=1
        if (r0[k]>meanr0):

            npo=npo+1
            sump=sump+(r0[k]-meanr0)**2
        else:
            nn=nn+1
            sumn=sumn+(r0[k]-meanr0)**2
    print("large error"+str(2*lerrorcount))
    stdp=np.sqrt(sump/float(npo))
    stdn=np.sqrt(sumn/float(nn))
    return [meanr0,(np.std(totresult,axis=0))[0],stdp,stdn]

timet=time.time()
swparr=[[100,1],[1000,1],[100,1],[10000,2],[1000,2],[100,2]]
for i in range(1):
    print(i)
    tpi=tpi0*swparr[i][1]
    lwset=swparr[i][0]
    results = swp_lw(lwset)
    mean_arr=results[0]
    std_arr=results[1]
    std_arrp=results[2]
    std_arrn=results[3]

    print(bucarr)
    print("*&^%")
    if(swparr[i][1]==1):
        print("pi pulse")
    else:
        print("2pi pulse")
    print("FWHM=",lwset/1000,"kHz")
    print("P[0]:",mean_arr)
    print("P[1]:",1-mean_arr)
    print("Standard Deviation:",std_arr)
    if(swparr[i][1]==1):
        print("Standard Deviation P:",std_arrp)
        print("Standard Deviation N:",std_arrn)
    else:
        print("Standard Deviation P:",std_arrn)
        print("Standard Deviation N:",std_arrp)
    print("-------")
#For parallel use below
##num_cores = 1
##results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
##results = np.array(results)
##print("Averaged population for each state:")
##print("y[0]="+str(np.mean(results[:,0])) + "+-" + str(np.std(results[:,0])/np.sqrt(nrun)))
##print("y[1]="+str(np.mean(results[:,1]))+ "+-" + str(np.std(results[:,1])/np.sqrt(nrun)))

