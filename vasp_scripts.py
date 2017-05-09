from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter
from vasp import Vasp
from ase.optimize import QuasiNewton
import os
#Need to handle setting the pseudopotential directory, probably in the submission config if it stays constant? (vasp_qadapter.yaml)
def runVasp(fname_in,fname_out,vaspflags,npar=4):
    fname_in=str(fname_in)
    fname_out=str(fname_out)

    #read the input atoms object
    atoms=read(str(fname_in))
    
    #update vasprc file to set mode to "run" to ensure that this runs immediately
    Vasp.vasprc(mode='run')
    #set ppn>1 so that it knows to do an mpi job, the actual ppn will guessed by Vasp module
    Vasp.VASPRC['queue.ppn']=2
    #if vaspflags['xc']=='beef-vdw':
        #if we're doing PBE, we're probably on a KNL node
        #vaspflags['NCORE']=1
    #else:
    #    NNODES=int(os.environ['SLURM_NNODES'])
    #    vaspflags['KPAR']=NNODES

    if 'PBS_SERVER' in os.environ and os.environ['PBS_SERVER']=='gilgamesh.cheme.cmu.edu':
        #We're on gilgamesh
        vaspflags['NPAR']=4
    elif 'SLURM_CLUSTER_NAME' in os.environ and os.environ['SLURM_CLUSTER_NAME']=='arjuna':
        #We're on arjuna
        if os.environ['CUDA_VISIBLE_DEVICES']!='NoDevFiles':
            #We have a GPU job on arjuna
            vaspflags['NCORE']=1
            vaspflags['KPAR']=16
            vaspflags['NSIM']=8
        else:
            #We're running CPU only
            vaspflags['NCORE']=4
            vaspflags['KPAR']=2
    elif 'SLURM_CLUSTER_NAME' in os.environ and	os.environ['SLURM_CLUSTER_NAME']=='cori':
        #We're on cori
        if os.environ['CRAY_CPU_TARGET']=='haswell':
            #We're on a haswell CPU node
            NNODES=int(os.environ['SLURM_NNODES'])
            vaspflags['KPAR']=NNODES
        elif os.environ['CRAY_CPU_TARGET']=='knl':
            vaspflags['NCORE']=1

    pseudopotential=vaspflags['pp_version']
    os.environ['VASP_PP_PATH']=os.environ['VASP_PP_BASE']+'/'+str(pseudopotential)+'/'

    #set up the calculation and run
    calc=Vasp('./',atoms=atoms,**vaspflags)
    calc.read_results()
        
    #Get the final trajectory
    atomslist=calc.traj
    finalimage=atomslist[-1]

    #Write a traj file for the optimization
    tj=TrajectoryWriter('all.traj','a')
    for atoms in atomslist:
        print('writing trajectory file!')
        print(atoms)
        tj.write(atoms)
    tj.close() 

    #Write the final structure
    finalimage.write(fname_out)
    
    #Write a text file with the energy
    with open('energy.out','w') as fhandle:
        fhandle.write(str(finalimage.get_potential_energy()))

    return str(atoms),open('all.traj','r').read().encode('hex'),finalimage.get_potential_energy()

def runVaspASEOptimizer(fname_in,fname_out,vaspflags):
    fname_in=str(fname_in)
    fname_out=str(fname_out)

    #read the input atoms object
    atoms=read(str(fname_in))

    #set ibrion=-1 and nsw=0
    vaspflags['ibrion']=-1
    vaspflags['nsw']=0

    #update vasprc file to set mode to "run" to ensure that this runs immediately
    Vasp.vasprc(mode='run')
    #set ppn>1 so that it knows to do an mpi job, the actual ppn will guessed by Vasp module
    Vasp.VASPRC['queue.ppn']=2
    vaspflags['NPAR']=4
    #set up the calculation and run
    calc=Vasp('./',atoms=atoms,**vaspflags)
    #calc.update()
    #calc.read_results()

    qn=QuasiNewton(atoms,logfile='relax.log',trajectory='relax.traj')
    qn.run(fmax=vaspflags['ediffg'] if 'ediffg' in vaspflags else 0.05)

    atoms.write(fname_out)

    #Write a text file with the energy
    with open('energy.out','w') as fhandle:
        fhandle.write(str(atoms.get_potential_energy()))

    return str(atoms),open('relax.traj','r').read().encode('hex'),atoms.get_potential_energy()



