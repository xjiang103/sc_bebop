import matplotlib.pyplot as plt
import numpy as np
print(3/5)
f=open("f3qs_0218.txt","r")
testarr="1 2 3"
a,b,c=testarr.split()
xa=[]
ya=[]
yqsa=[]

r=f.readline()
for i in range(10):
    r=f.readline()

    x,y,yqs=r.split(' ')
    print(x)
    #y=rtmp[1]
    #yqs=rtmp[2]
    xa.append(float(x))
    ya.append(float(y))
    yqsa.append(float(yqs))

    #print(float(y))

f.close()
plt.plot(xa,ya,lw=2,label="Simulation")
plt.plot(xa,yqsa,lw=2,label="Quasi-Static")
##for i in range(int(116)):
##    plt.plot([xa[i],xa[i]],[ya[i]-sdna[i],ya[i]+sdpa[i]],'r')

plt.yscale('log')
plt.legend()
plt.show()

