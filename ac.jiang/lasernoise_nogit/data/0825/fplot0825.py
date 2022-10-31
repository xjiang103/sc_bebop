import matplotlib.pyplot as plt
import matplotlib
tpnum=2
matplotlib.rcParams.update({'font.size': 16})
filestr="fmaxswp_0820_"+str(tpnum)+".txt"
f=open(filestr,"r")
xa=[]
ya=[]
siga=[]
for i in range(25):
    r=f.readline()
    x,y,sig=r.split()
    print(str(i)+" "+x+" "+y+" "+sig)
    xa.append(float(x)/1000)
    ya.append(float(y))
    siga.append(float(sig))
f.close()

tpnum=4
matplotlib.rcParams.update({'font.size': 16})
filestr="fmaxswp_0820_"+str(tpnum)+".txt"
f=open(filestr,"r")
xb=[]
yb=[]
sigb=[]
for i in range(25):
    r=f.readline()
    x,y,sig=r.split()
    print(str(i)+" "+x+" "+y+" "+sig)
    xb.append(float(x)/1000)
    yb.append(float(y))
    sigb.append(float(sig))
f.close()

tpnum=6
matplotlib.rcParams.update({'font.size': 16})
filestr="fmaxswp_0820_"+str(tpnum)+".txt"
f=open(filestr,"r")
xc=[]
yc=[]
sigc=[]
for i in range(25):
    r=f.readline()
    x,y,sig=r.split()
    print(str(i)+" "+x+" "+y+" "+sig)
    xc.append(float(x)/1000)
    yc.append(float(y))
    sigc.append(float(sig))
f.close()

plt.plot(xa,ya,'o-',label="numerics-2π")
plt.plot(xa,siga,label="quasistatic-2π")
plt.plot(xb,yb,'o-',label="numerics-4π")
plt.plot(xb,sigb,label="quasistatic-4π")
plt.plot(xc,yc,'o-',label="numerics-6π")
plt.plot(xc,sigc,label="quasistatic-6π")

plt.legend(loc="lower right")
plt.yscale('log')
if (tpnum%2==0):
    plt.ylim(10**(-8),10**(-3))
plt.xlabel("Linewidth/kHz")
plt.ylabel('error')
plt.title("T=2,4,6π/Ω")
plt.show()

