import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import time
import random
from scipy.integrate import solve_ivp
from scipy import stats
nrun=10
units=1e0
#rabi parameters
omega1=2*np.pi*(44.72)*(units)
omega2=2*np.pi*(44.72)*(units)
delta1=2*np.pi*(1000)*(units)
delta2=delta1*(1-np.sqrt(1+(omega1**2-omega2**2)/(2*delta1**2)))-delta1
delta=delta1-delta2
deltasum=delta1+delta2
delta1prime=deltasum+(omega1**2)/delta
delta2prime=deltasum-(omega2**2)/delta
deltaplus=(delta1prime+delta2prime)/2
deltaminus=(delta1prime-delta2prime)/2
omegar=omega1*omega2/delta
omegaprime=np.sqrt(omegar**2+deltaplus**2)
print("δ="+str(delta/(2*np.pi*units)))
print("ΩR="+str(omegar/(2*np.pi*units)))
print("Ωprime="+str(omegaprime/(2*np.pi*units)))
print(delta)
print(omega1)
print(omega2)
