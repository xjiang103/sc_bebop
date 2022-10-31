import matplotlib.pyplot as plt
import numpy as np
print(3/5)
f=open("fswp16_1106.txt","r")
xa=[]
ya=[]
xarr=[]
yarr=[]
sdna=[]
sdpa=[]
xacc=0
yacc=0
yatot=0
for i in range(116):
    r=f.readline()
    x,y,sdn,sdp=r.split()
    xacc=xacc+float(x)
    yacc=yacc+float(y)
    
    if (i%1==0):
        xarr.append(xacc)
        yarr.append(yacc)
        xacc=0
        yacc=0
    xa.append(float(x))
    ya.append(float(y))
    sdna.append(float(sdn))
    sdpa.append(float(sdp))
    yatot=yatot+float(y)
    #print(float(y))
print(yatot/54)

plt.plot(xa,ya)
##for i in range(int(54)):
##    plt.plot([xa[i],xa[i]],[ya[i]-sdna[i],ya[i]+sdpa[i]],'r')
f.close()
plt.xlabel("fcenter/(Ω0/2π))")
plt.ylabel('error')
plt.title("2Pi pulse error")
plt.show()

