import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 24})
print(3/5)
tpi=2
filestr="hgswp_1031_"+str(tpi)+".txt"
f=open(filestr,"r")
xa=[]
ya=[]
stdpa=[]
stdna=[]
    
for i in range(5):
    r=f.readline()
    x,y,stdp,stdn=r.split()
    xa.append(float(x))
    ya.append(float(y))
    stdpa.append(float(y)+float(stdp))
    stdna.append(float(y)-float(stdn))
plt.plot(xa,ya,'o-',label="Error")
plt.plot(xa,stdpa,'r',alpha=0.2,label="Uncertainty")
plt.plot(xa,stdna,'r',alpha=0.2)
#for i in range(5):
#    plt.plot([xa[i],xa[i]],[stdna[i],stdpa[i]],'r')
plt.fill_between(xa,stdpa,stdna,color='crimson',alpha=0.1)
##for i in range(int(116)):
##    plt.plot([xa[i],xa[i]],[ya[i]-sdna[i],ya[i]+sdpa[i]],'r')
f.close()
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Fractional Power")
plt.ylabel('Error')
plt.title("Error at time "+str(tpi)+"π/Ω")

plt.legend(loc="right")

plt.show()

