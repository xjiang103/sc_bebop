import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
f=open("f3swp2_0811_cat.txt","r")
#f=open("fswp_0706_2.txt","r")
xa=[]
ya=[]
for i in range(120):
    r=f.readline()
    x,y=r.split()
    print(str(i)+" "+x+" "+y)
    xa.append(float(x))
    ya.append(float(y))


plt.plot(xa,ya,'o-')
plt.xlabel("fcenter/(Ω0/2π))")
plt.ylabel('Error')
plt.title("2-photon, t=2π/Ω")
plt.show()

