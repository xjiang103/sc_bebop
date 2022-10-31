
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy import stats

#number of runs for averaging
nrun=100

#rabi parameters
omega_0=2*math.pi*(1)*(1e6)
tpi0=1*math.pi/(omega_0)
tpi=tpi0
#frequency domain sample parameters
fmax=0.01*omega_0/(2*math.pi)
nf=1000
fmin=fmax/nf
df=fmin

#linewidth sweep parameters(currently not sweeping. Linewidth is set to
#equal to lwmax
lwset=100
lwfactor=1
#pre-calculating some constants

pikdf_arr= np.zeros(nf)
s_nu_arr= np.zeros(nf) 
for k in range(nf):
    pikdf_arr[k] = np.pi*(k+1)*df

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
    for count in range(nrun):
        res.append(run_job())
    totresult=np.array(res)
    r0=totresult[:,0]
    meanr0=np.mean(r0)
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
x_fmax=[]
y1_fmax=[]
sp_fmax=[]
sn_fmax=[]
y2_fmax=[]
y3_fmax=[]
s2_fmax=[]
for i in range(10):
#    tpi=tpi0*swparr[i][1]
    print(i)
    tpi=9*tpi0
    lwset=10000*(i+1)
    res=swp_lw(lwset)
    y1_fmax.append(res[0])
    sp_fmax.append(res[1])
    sn_fmax.append(res[2])
#    lwfactor=1/((i+1)*0.1)
#    res=swp_lw(lwset)
#    y2_fmax.append(res[0])
#    s2_fmax.append(res[1])
    y2_fmax.append((3/16)*(4.5)**2*(np.pi)**2*(lwset*fmax/np.pi)**2/((omega_0/(2*math.pi))**4))
    y3_fmax.append(0.5*(lwset*fmax/np.pi)/((omega_0/(2*math.pi))**2))
    x_fmax.append(lwset/1000)
    print(lwset/1000)
plt.plot(x_fmax,y1_fmax,label="Simulation")

for i in range(10):
    plt.plot([x_fmax[i],x_fmax[i]],[y1_fmax[i]-sn_fmax[i],y1_fmax[i]+sp_fmax[i]],'r')
plt.plot(x_fmax,y2_fmax,label="Quasi static 1")
##plt.plot(x_fmax,y3_fmax,label="New quasi static")
plt.yscale('log')
plt.xlabel("lw/kHz")
plt.ylabel("error")
plt.title("Error at t=9π/Ω")
plt.legend(loc='upper right')
plt.show()
#For parallel use below
##num_cores = 1
##results = Parallel(n_jobs=num_cores,backend="loky")(delayed(run_job)() for count in range(nrun))
##results = np.array(results)
##print("Averaged population for each state:")
##print("y[0]="+str(np.mean(results[:,0])) + "+-" + str(np.std(results[:,0])/np.sqrt(nrun)))
##print("y[1]="+str(np.mean(results[:,1]))+ "+-" + str(np.std(results[:,1])/np.sqrt(nrun)))

