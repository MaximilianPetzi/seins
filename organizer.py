Nsims=10 #number of simulations per parameter config

#to make many simulations at once with the parameters in paramfile.npy
#paramfile is set by set_params.py
import os 
import time
from termcolor import colored
import numpy as np
os.system("rm -r -f stop")
os.system("mkdir error_org")
while True: 
    maxcount=Nsims  #number of simulations that should run simultaneously
    np.save("maxcount.npy",maxcount) #maxcount not needed, delete!

    content=np.load("paramfile.npy",allow_pickle=True)
    todo=content.item().get("todo")#simulations to do, starting with todo[0]
    done=content.item().get("done")#simulations done, the latest simulation is done[-1]
    print("len todo=",len(todo))
    if len(todo)==0:print("---nothing todo, finished---"); break
    print(colored("---new param config---","red"))

    #take first element of todo list and move it to done stack
    newpar=todo[0]
    done=np.concatenate([done,[todo[0]]])
    todo=todo[1:]

    dirname=str(newpar[0])+"_"+str(newpar[1])+"_"+str(newpar[2])+"_"+str(newpar[3])
    
    os.system("rm -r -f error_org/"+dirname)#clear previous folder
    os.system("mkdir error_org/"+dirname)


    lasttime=time.time()    #time of last update, updated in every iteration of jiggling.py
    np.save("lasttime.npy", lasttime)

    os.system("bash simulations.sh")
    print(dirname)
    print(colored(done, "red"))
    while True:
        time.sleep(20)#increase this to 60 seconds
        #print("checking if idle and if not finished, entering weird loop")
        while True:
            try:
                lasttime=np.load("lasttime.npy")
                break
            except:
                time.sleep(0.2)

        #lasttime=time.time()

        #check if every process has stopped but the simulations aren't complete yet. 
        if time.time()-lasttime>20 and not os.path.isfile("error_org/"+dirname+"/"+str(Nsims)+"error.npy"):
            # then start another batch of simulations
            print(colored("start new sims","red"))
            os.system("bash simulations.sh")
            time.sleep(30)  #wait (long enough to let first simulation start||
                            # but long enough to not occupy to many cores at once) 
        #test if Nsim simulations are complete:
        if os.path.isfile("error_org/"+dirname+"/"+str(Nsims)+"error.npy"):
            print(colored("files complete","red"))
            break
        #print("error_org/"+dirname+"/"+str(Nsims)+"error.npy is still missing")

    content={"todo":todo,"done":done}
    np.save("paramfile.npy",content)
