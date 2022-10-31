import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 16})
data = np.loadtxt("ln1_0804.txt")
lw_arr=data[:,0]/1000
print(lw_arr)
t2_arr=data[:,1]*1000000
lwfit_arr=np.linspace(0,100,1000)

def func(x,a):
    return a*(1/pow(x,1))

popt,pcov=curve_fit(func,lw_arr,t2_arr)
aaa=popt[0]
t2fit_arr=func(lwfit_arr,aaa)

print(aaa)
print(1000/(2*np.pi))
theoconst=1000/(2*np.pi)
t2theo_arr=func(lwfit_arr,theoconst)

plt.clf()
plt.plot(lw_arr,t2_arr,'ro',label="T2,numerics")
#plt.plot(lwfit_arr,t2fit_arr,label="Fitting numerics to 1/x function")
plt.plot(lwfit_arr,t2theo_arr,label="Theory")

plt.xlabel("Linewidth/kHz")
plt.ylabel("T2/μs")
plt.title("T2-Linewidth")
#plt.xlim([0,110])
plt.legend(loc="upper right")
plt.ylim([0,20])
plt.show()
plt.savefig("pops1.png",dpi=300)
