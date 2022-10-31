import matplotlib.pyplot as plt
import matplotlib
import numpy as np
print(3/5)
tpi=2
matplotlib.rcParams.update({'font.size': 24})
filestr="fswp_1103_"+str(tpi)+".txt"
f=open(filestr,"r")
xa=[]
ya=[]
ta=[]
spa=[]
sna=[]
omegar=2*np.pi*(1e6)
h_g=1100
f_g0=234*(1e3)
sigma_g=1.4*(1e3)


for i in range(17):
    r=f.readline()
    x,y,sp,sn=r.split()
    print(str(i)+" "+x+" "+y)
    xa.append(float(x))
    ya.append(float(y))
    spa.append(float(y)+float(sp))
    sna.append(float(y)-float(sn))
##    if(i==40):
##        x=1
    fg=float(x)*(1e6)
    omegag=2*np.pi*fg
    sg=np.sqrt(8*np.pi)*sigma_g*h_g/(f_g0**2)
    #print(2*omegag**2*sg/(omegar**2))
    N=1/1
    print(omegag/omegar)
    et=2*omegag**2*sg*(1/omegar**2)*((np.cos(0.5*np.pi*omegag/omegar))**2*\
        (1-(-1)**(2*N)*np.cos(2*np.pi*N*omegag/omegar))/(4*(omegar**2-omegag**2)**2/omegar**4)+\
        (np.sin(0.5*np.pi*omegag/omegar))**2*2*np.pi*N*(1+2*np.pi*N)/32)
    print(sg)
    ta.append(et)


plt.plot(xa,ya,'o',label="numerics")
plt.plot(xa,ta,label="theory")
plt.plot(xa,spa,'r',alpha=0.2,label="Uncertainty")
plt.plot(xa,sna,'r',alpha=0.2)
plt.fill_between(xa,spa,sna,color='crimson',alpha=0.1)

plt.xlabel("fg/(Ω0/2π))")
plt.title(str(tpi)+"pi Error,sweeping fg")
plt.legend(loc="lower right")
plt.yscale('log')
plt.ylabel('error')
plt.show()

