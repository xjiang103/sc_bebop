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
xxx1=[10000,5000,1000,200,50]

yyy1=[t2_arr1[3],t2_arr2[3],t2_arr3[3],t2_arr4[3],t2_arr5[3]]

print(yyy1)

plt.plot(xxx1,yyy1)
plt.xlabel("Bandwidth/kHz")
plt.ylabel("T2/μs")
plt.title("T2-bandwidth")
plt.xscale('log')
plt.show()


