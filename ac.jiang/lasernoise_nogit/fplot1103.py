import matplotlib.pyplot as plt
import matplotlib
tpnum=2
matplotlib.rcParams.update({'font.size': 24})
filestr="fmaxswp_1103_"+str(tpnum)+".txt"
f=open(filestr,"r")
xa=[]
ya=[]
spa=[]
sna=[]
siga=[]
for i in range(20):
    r=f.readline()
    x,y,sig,sp,sn=r.split()
    print(str(i)+" "+x+" "+y+" "+sig)
    xa.append(float(x)/1000)
    ya.append(float(y))
    siga.append(float(sig))
    spa.append(float(y)+float(sp))
    sna.append(float(y)-float(sn))
f.close()

plt.plot(xa,ya,'o',label="numerics")
plt.plot(xa,siga,label="quasistatic")
plt.plot(xa,spa,'r',alpha=0.2,label="Uncertainty")
plt.plot(xa,sna,'r',alpha=0.2)
plt.fill_between(xa,spa,sna,color='crimson',alpha=0.1)
plt.legend(loc="lower right")
plt.yscale('log')
if (tpnum%2==0):
    plt.ylim(10**(-8),10**(-4))
plt.xlabel("Linewidth/kHz")
plt.ylabel('Error')
plt.title("T="+str(tpnum)+"π/Ω")
plt.show()

