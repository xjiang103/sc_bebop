import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
f=open("ln32_0712_1.txt","r")
xa=[]
ya=[]
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
f.readline()
for i in range(32):
    r1=f.readline()
    x=f.readline()
    y=f.readline()
    r2=f.readline()
    print(str(i)+" "+x+" "+y)
    xa.append(float(x))
    ya.append(float(y))


plt.plot(xa,ya,'o-')
plt.xlabel("fcenter/(Ω0/2π))")
plt.ylabel('error')
plt.title("2-photon, t=1π/Ω")
plt.show()

