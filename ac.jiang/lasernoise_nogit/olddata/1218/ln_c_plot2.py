import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 16})
data1 = np.loadtxt("ln_c_2.txt")
lw_arr1=data1[:,0]/1000
t2_arr1=data1[:,1]*1000000

data2 = np.loadtxt("ln2_1218.txt")
lw_arr2=data2[:,0]/1000
t2_arr2=data2[:,1]*1000000

data3 = np.loadtxt("ln3_1218.txt")
lw_arr3=data3[:,0]/1000
t2_arr3=data3[:,1]*1000000

data4 = np.loadtxt("ln4_1218.txt")
lw_arr4=data4[:,0]/1000
t2_arr4=data4[:,1]*1000000

data5 = np.loadtxt("ln5_1218.txt")
lw_arr5=data5[:,0]/1000
t2_arr5=data5[:,1]*1000000


lwfit_arr=np.linspace(0,100,1000)

def func(x,a):
    return a*(1/pow(x,1))

popt,pcov=curve_fit(func,lw_arr1,t2_arr1)
aaa=popt[0]
t2fit_arr=func(lwfit_arr,aaa)

plt.clf()
plt.plot(lw_arr1,t2_arr1,label="BW=10MHz")
plt.plot(lw_arr2,t2_arr2,label="BW=5MHz")
plt.plot(lw_arr3,t2_arr3,label="BW=1MHz")
plt.plot(lw_arr4,t2_arr4,label="BW=200kHz")
plt.plot(lw_arr5,t2_arr5,label="BW=50kHz")

plt.legend(loc='upper right')

plt.plot(lwfit_arr,t2fit_arr,'g')

plt.xlabel("Linewidth/kHz")
plt.ylabel("T2/μs")
plt.title("T2-Linewidth")
#plt.xlim([0,110])
plt.ylim([0,20])
plt.show()
plt.savefig("pops1.png",dpi=300)
