from sympy.physics.wigner import wigner_6j
a=wigner_6j(1/2,7/2,4,4,1,1/2)
print(a)

from sympy.physics.wigner import clebsch_gordan
b=clebsch_gordan(4,1,4,-4,0,-4)
print(b)
