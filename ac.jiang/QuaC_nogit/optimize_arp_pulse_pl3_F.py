from scipy.optimize import minimize
import subprocess
import numpy as np
import uuid
import os
from qutip import *
from qutip.qip.operations import snot,phasegate
import argparse


parser = argparse.ArgumentParser(description='Optimize Pulses')
parser.add_argument('-b','--b_couple', help='Rydberg Coupling Strength', required=True, type=float)
parser.add_argument('-ist','--initial_state', help='Variation in Rydberg Coupling Strength', required=True, type=int)
args = vars(parser.parse_args())
b = args["b_couple"]
ist=args["initial_state"]
print("initial state is "+str(ist))
ddfac=1
typestr="ccz"


unique_file = str(uuid.uuid4())[0:8]
file_name = "dm_"+unique_file+".dat" #Allow us to run in parallel
params = [23]
#b=100
f=open("op_3_arp_par_F2.txt","a")
f.write('\n')
f.write("initial state="+str(ist)+'\n')
f.write(typestr+' ')

def ia(init_state):
    state_arr=[]
    for k in range(len(init_state)):
        if (init_state[k]=='0'):
            state_arr.append(basis(2,0))
        elif (init_state[k]=='1'):
            state_arr.append(basis(2,1))
    state=tensor(state_arr[0],state_arr[1],state_arr[2])
    return state
init_arr=[]
str_arr=[]
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            initstr=""
            if (j1==0):
                initstr=initstr+'0'
            elif (j1==1):
                initstr=initstr+'1'
            if (j2==0):
                initstr=initstr+'0'
            elif (j2==1):
                initstr=initstr+'1'
            if (j3==0):
                initstr=initstr+'0'
            elif (j3==1):
                initstr=initstr+'1'
            str_arr.append(initstr)
            init_arr.append(ia(initstr))
ave_state=0
for j in range(8):
    ave_state+=(1/np.sqrt(8))*init_arr[j]
init_arr.append(ave_state)
str_arr.append("xxx")

init_state=init_arr[ist]

print("init_state is "+str_arr[ist])
print("init state:")
print(init_state)

def fun_sp(params,final_run=None):


    #Run QuaC
    try:
        output = subprocess.check_output(["./na_3_par_3lvl2_F2","-ts_rk_type","5bs","-ts_rtol","1e-8","-ts_atol","1e-8","-n_ens","-1",
                                          "-pulse_type","ARP","-file",file_name,
                                          "-bitstr",str_arr[ist],
                                          "-b_term",str(b),
                                          "-delta",str(params[0]),
                                          "-pulse_length",str(params[1]),
                                          "-dd_fac",str(ddfac)])
    except:
        pass

    #Read in the QuaC DM
    dm = Qobj(np.loadtxt(file_name).view(complex),dims=[[2,2,2],[2,2,2]])
    #Remove file
    os.remove(file_name)

    #QUTIP to get perfect circuit
    res = minimize(qutip_phase,[0,0,0],method="COBYLA",args=(dm))

    fid = 1-res.fun
    print(fid)
    if(final_run):
        print("Phase: ",res.x)
        f.write(str(res.x[0])+' ')
        f.write(str(res.x[1])+' ')
        f.write(str(res.x[2])+' ')
    return 1-fid
def print_callback(xs):
    print(xs)
def qutip_phase(params,dm):
    #define cz_arp and czz arp
    ccz_arp = Qobj([[1,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0],[0,0,0,-1,0,0,0,0],[0,0,0,0,-1,0,0,0],[0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,-1]],dims=[[2,2,2],[2,2,2]])
    czz_arp = Qobj([[1,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],dims=[[2,2,2],[2,2,2]])
    
    state=init_state
    #Apply phase gates with parameters that we are optimizing
    state = tensor(phasegate(params[0]),qeye(2),qeye(2))*state
    state = tensor(qeye(2),phasegate(params[1]),qeye(2))*state
    state = tensor(qeye(2),qeye(2),phasegate(params[2]))*state

    #Now apply cz_arp
    state = ccz_arp*state

    #Get fidelity wrt quac dm
    fid = fidelity(dm,state)

    return 1-fid

def fun_arp(delta):
    #NOT COMPLETED!
    return 1-fid


print("Optimizing ARP for b = ",str(b))
print("Optimizing Delta, T, and phases")
f.write(str(ddfac)+' ')
#f.write(str(b)+' ')
f.write("Delta_T_phases for b="+str(b)+' ')
default_sp_params = [-0.5,0.2]
default_sp_params = [23,0.54]
res = minimize(fun_sp,default_sp_params,method="nelder-mead",callback=print_callback)

#get the optimal phases
fun_sp(res.x,True)

print("Final Fidelity: ",str(1-res.fun))
f.write(str(1-res.fun)+' ')
print("Final Params: ",str(res.x))
f.write(str(res.x[0])+' ')
f.write(str(res.x[1])+' ')
f.write('\n')
#Final Fidelity:  0.9997463238664505
f.close()

