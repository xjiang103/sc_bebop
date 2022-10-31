import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 12})
fig = plt.figure()
fig.set_size_inches(3.375,3.375*2)

gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)

h0=200

print(3/5)
tpi=1

filestr="intswp1_1.txt"
f=open(filestr,"r")
xa=[]
ya=[]
ta=[]
spa=[]
sna=[]
omegar=2*np.pi*(1e6)
h_g=200


for i in range(20):
    r=f.readline()
    x,y,sp,sn=r.split()
    print(str(i)+" "+x+" "+y)
    xa.append(0.01*(i+1))
    ya.append(float(y))
    spa.append(float(y)+float(sp))
    sna.append(float(y)-float(sn))
##    if(i==40):
##        x=1
    fg=float(x)*(1e6)

    #print(2*omegag**2*sg/(omegar**2))
    N=1/2
    et=N**2*(np.pi)**2*(0.02*(i+1))**2/4
    ta.append(1*et)


axs[0].plot(xa,ya,'o-',label="Numerics",color='red')
axs[0].plot(xa,ta,label="Theory",color='blue')
axs[0].plot(xa,spa,'r',alpha=0.2,label="Uncertainty")
axs[0].plot(xa,sna,'r',alpha=0.2)
axs[0].fill_between(xa,spa,sna,color='crimson',alpha=0.1)
axs[0].legend(loc="lower right", prop={'size': 10})
axs[0].set_yscale('log')
axs[0].text(0.1, 0.9, 'a', horizontalalignment='center',
     verticalalignment='center', transform=axs[0].transAxes)
f.close()

tpi=2
filestr="intswp1_2.txt"
f=open(filestr,"r")
xa=[]
ya=[]
ta=[]
spa=[]
sna=[]
omegar=2*np.pi*(1e6)
h_g=200


for i in range(20):
    r=f.readline()
    x,y,sp,sn=r.split()
    print(str(i)+" "+x+" "+y)
    xa.append(0.01*(i+1))
    ya.append(float(y))
    spa.append(float(y)+float(sp))
    sna.append(float(y)-float(sn))
##    if(i==40):
##        x=1
    fg=float(x)*(1e6)

    #print(2*omegag**2*sg/(omegar**2))
    N=1/1
    et=N**2*(np.pi)**2*(0.02*(i+1))**2/4
    ta.append(1*et)


axs[1].plot(xa,ya,'o-',label="Numerics",color='red')
axs[1].plot(xa,ta,label="Theory",color='blue')
axs[1].plot(xa,spa,'r',alpha=0.2,label="Uncertainty")
axs[1].plot(xa,sna,'r',alpha=0.2)
axs[1].fill_between(xa,spa,sna,color='crimson',alpha=0.1)
axs[1].legend(loc="lower right", prop={'size': 10})
axs[1].set_yscale('log')
axs[1].text(0.1, 0.9, 'b', horizontalalignment='center',
     verticalalignment='center', transform=axs[1].transAxes)
f.close()

plt.xlabel("RIN")
plt.ylabel('Error')
fig.show()

plt.savefig('int_1pw.pdf', bbox_inches='tight')

