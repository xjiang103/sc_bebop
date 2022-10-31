import matplotlib.pyplot as plt
print(3/5)
f=open("fswp5_1023.txt","r")
xa=[]
ya=[]
xarr=[]
yarr=[]
sdna=[]
sdpa=[]
xacc=0
yacc=0
for i in range(54):
    r=f.readline()
    x,y,sdn,sdp=r.split()
    print(str(i)+" "+x+" "+y)
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


plt.plot(xa,ya)
for i in range(int(54)):
    plt.plot([xa[i],xa[i]],[ya[i]-sdna[i],ya[i]+sdpa[i]],'r')
plt.xlabel("fcenter/(Ω0/2π))")
plt.ylabel('error')
plt.show()

