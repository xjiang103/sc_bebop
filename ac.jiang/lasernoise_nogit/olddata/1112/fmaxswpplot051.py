import matplotlib.pyplot as plt
import numpy as np
print(3/5)
f=open("fswp05pi1_1112.txt","r")
xa=[]
ya=[]
xarr=[]
yarr=[]
sdna=[]
sdpa=[]
p1a=[]
p2a=[]
ffa=[]
xacc=0
yacc=0
yatot=0
for i in range(116):
    r=f.readline()
    x,y,sdn,sdp,y2,p1,p2,ff=r.split()
    print(y)
    xacc=xacc+float(x)
    yacc=yacc+float(y)
    
    if (i%1==0):
        xarr.append(xacc)
        yarr.append(yacc)
        xacc=0
        yacc=0
    xa.append(float(x))
    ya.append(float(y))
    p1a.append(float(p1))
    p2a.append(float(p2)-np.pi/2)
    ffa.append(float(ff))

    sdna.append(float(sdn))
    sdpa.append(float(sdp))
    yatot=yatot+float(y)
    #print(float(y))
print(yatot/54)

plt.plot(xa,p2a)
##for i in range(int(116)):
##    plt.plot([xa[i],xa[i]],[ya[i]-sdna[i],ya[i]+sdpa[i]],'r')
f.close()
plt.xlabel("fg/(Ω0/2π))")
plt.ylabel('phase')
plt.title("Pi/2 pulse, phase of |e>-Pi/2")
plt.show()

