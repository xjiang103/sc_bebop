from scipy.optimize import minimize
import subprocess
import numpy as np
import uuid
import os
from qutip import *
from qutip.qip.operations import snot,phasegate
import argparse


##parser = argparse.ArgumentParser(description='Optimize Pulses')
##parser.add_argument('-b','--b_couple', help='Rydberg Coupling Strength', required=True, type=float)
##args = vars(parser.parse_args())
##b = args["b_couple"]
ddfac=0
typestr="czz"


unique_file = str(uuid.uuid4())[0:8]
file_name = "dm_"+unique_file+".dat" #Allow us to run in parallel
params = [-0.5,0.2]

f=open("op_3_par_0929.txt","a")
f.write('\n')
f.write(typestr+' ')
def fun_sp(params,final_run=None):


    #Run QuaC
    try:
        output = subprocess.check_output(["./na_3_par_3lvl2","-ts_rk_type","5bs","-ts_rtol","1e-8","-ts_atol","1e-8","-n_ens","-1",
                                          "-pulse_type","SP","-file",file_name,
                                          "-b_term",str(params[2]),
                                          "-delta",str(params[0]),
                                          "-deltat",str(params[1]),
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

def qutip_phase(params,dm):
    #define cz_arp and czz arp
    ccz_arp = Qobj([[1,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0],[0,0,0,-1,0,0,0,0],[0,0,0,0,-1,0,0,0],[0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,-1,0],[0,0,0,0,0,0,0,-1]],dims=[[2,2,2],[2,2,2]])
    czz_arp = Qobj([[1,0,0,0,0,0,0,0],[0,-1,0,0,0,0,0,0],[0,0,-1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],dims=[[2,2,2],[2,2,2]])
    
    #Get two hadamard state
    state = tensor(snot(),snot(),snot())*tensor(basis(2,1),basis(2,1),basis(2,1))

    #Apply phase gates with parameters that we are optimizing
    state = tensor(phasegate(params[0]),qeye(2),qeye(2))*state
    state = tensor(qeye(2),phasegate(params[1]),qeye(2))*state
    state = tensor(qeye(2),qeye(2),phasegate(params[2]))*state

    #Now apply cz_arp
    state = czz_arp*state
    #print("trace=",str(np.trace(dm)))
    #Get fidelity wrt quac dm
    fid = fidelity(dm,state)

    return 1-fid

def fun_arp(delta):
    #NOT COMPLETED!
    return 1-fid


#print("Optimizing SP for b = ",str(b))

f.write(str(ddfac)+' ')
#f.write(str(b)+' ')
default_sp_params = [-0.5,0.2]
default_sp_params = [-0.5,0.2165,60]
res = minimize(fun_sp,default_sp_params,method="nelder-mead")

#get the optimal phases
fun_sp(res.x,True)
print("Final Fidelity: ",str(1-res.fun))
f.write(str(1-res.fun)+' ')
print("Final Params: ",str(res.x))
f.write(str(res.x[0])+' ')
f.write(str(res.x[1])+' ')
f.write(str(res.x[2])+' ')
f.write('\n')
#Final Fidelity:  0.9997463238664505
f.close()
