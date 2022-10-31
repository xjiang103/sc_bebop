import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
f=open("f3swp2_0723.txt","r")
xa=[]
ya=[]
for i in range(80):
    print(i)
    r=f.readline()
    x,y=r.split()
    xa.append(float(x))
    ya.append(float(y))


plt.plot(xa,ya,'o-')
plt.xlabel("fcenter/(Ω0/2π))")
plt.ylabel('error')
plt.title("2-photon, t=2π/Ω")
plt.show()

