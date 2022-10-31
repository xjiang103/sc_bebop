import matplotlib.pyplot as plt
import matplotlib
tpnum=1
matplotlib.rcParams.update({'font.size': 16})
filestr="fmaxswp_0802_"+str(tpnum)+".txt"
f=open(filestr,"r")
xa=[]
ya=[]
siga=[]
for i in range(20):
    r=f.readline()
    x,y,sig=r.split()
    print(str(i)+" "+x+" "+y+" "+sig)
    xa.append(float(x)/1000)
    ya.append(float(y))
    siga.append(float(sig))
f.close()

plt.plot(xa,ya,'o',label="numerics")
plt.plot(xa,siga,label="quasistatic")

plt.legend(loc="lower right")
plt.yscale('log')
if (tpnum%2==0):
    plt.ylim(10**(-9),10**(-5))
plt.xlabel("Linewidth/kHz")
plt.ylabel('error')
plt.title("T="+str(tpnum)+"π/Ω")
plt.show()

