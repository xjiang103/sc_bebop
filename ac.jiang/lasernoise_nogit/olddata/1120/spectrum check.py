import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
nf=10
lw=1000
df=0.01
dt=0.0001/nf
pikdf_arr= np.zeros(nf)
s_nu_arr= np.zeros(nf) 
def s_nu(f,lw):
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
    snu=lw/(f*f)
  
    return snu/(f*f)
def phigen(t):
    phase_arr=np.random.uniform(0,2*np.pi,size=nf)
    phitemp = np.sum(s_nu_arr*np.cos(pikdf_arr*t+phase_arr))
    return phitemp
for k in range(nf):
    s_nu_arr[k] = 2*np.sqrt(s_nu((k+1)*df,lw)*df)
for k in range(nf):
    pikdf_arr[k] = 2*np.pi*(k+1)*df

y_nu_arr= np.zeros(nf)
x1_arr=[]
for j in range(1):
    
    tarr=[]
    xarr=[]
    yyarr=[]
    for i in range(nf):
        xarr.append((i+1)*dt)
        tarr.append(phigen((i+1)*dt))
    yarr=fft(tarr)
    x1_arr=xarr
    for i in range(nf):
        yy=np.abs(yarr[i])
        y_nu_arr[i]=y_nu_arr[i]+yy/10000
plt.plot(x1_arr,tarr)
plt.show()

