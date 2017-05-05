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
    if vaspflags['xc']=='beef-vdw':
        #if we're doing PBE, we're probably on a KNL node
        vaspflags['NCORE']=1
    else:
        NNODES=int(os.environ['SLURM_NNODES'])
        vaspflags['KPAR']=NNODES
    #set up the calculation and run
    calc=Vasp('./',atoms=atoms,**vaspflags)
    calc.update()
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



