import matplotlib.pyplot as plt
import matplotlib
import numpy as np
print(3/5)
matplotlib.rcParams.update({'font.size': 16})
f=open("fswp_0706_1.txt","r")
xa=[]
ya=[]
ta=[]
omegar=2*np.pi*(1e6)
h_g=1100
f_g0=234*(1e3)
sigma_g=1.4*(1e3)


for i in range(80):
    r=f.readline()
    x,y=r.split()
    print(str(i)+" "+x+" "+y)
    xa.append(float(x))
    ya.append(float(y))
##    if(i==40):
##        x=1
    fg=float(x)*(1e6)
    omegag=2*np.pi*fg
    sg=np.sqrt(8*np.pi)*sigma_g*h_g/(f_g0**2)
    #print(2*omegag**2*sg/(omegar**2))
    N=1/2
    print(omegag/omegar)
    et=2*omegag**2*sg*(1/omegar**2)*((np.cos(0.5*np.pi*omegag/omegar))**2*\
        (1-(-1)**(2*N)*np.cos(2*np.pi*N*omegag/omegar))/(4*(omegar**2-omegag**2)**2/omegar**4)+\
        (np.sin(0.5*np.pi*omegag/omegar))**2*2*np.pi*N*(1+2*np.pi*N)/32)
    print(sg)
    ta.append(et)


plt.plot(xa,ya,label="numerics")
plt.plot(xa,ta,label="theory")
plt.xlabel("fg/(Ω0/2π))")
plt.title("1pi Error,sweeping fg")
plt.legend(loc="lower right")
plt.yscale('log')
plt.ylabel('error')
plt.show()

