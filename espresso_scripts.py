import sys
import os
    
from ase import Atoms
from ase import *
from ase.optimize import QuasiNewton,BFGS
from numpy import array
from ase.calculators.neighborlist import NeighborList
from ase.constraints import FixAtoms
from ase.io import read,write
from espresso import espresso
import numpy as np
from ase.lattice.surface import add_adsorbate
from ase import Atoms
import ase
from ase.constraints import FixAtoms, Hookean
from ase.optimize.minimahopping import MinimaHopping
#from basc.basc import BASC
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from fireworks_helper_scripts import atomToString

import math
import os,sys,pickle
from copy import deepcopy

def relaxAtoms(fname_in,fname_out,cutoffs=500.):
    #This script relaxes the atoms
    convergence={'energy':0.0001,
               'mixing':0.1,
               'nmix':10,
               'maxsteps':500,
               'diag':'david'
                }

    dipole = {'status':False}

    atoms=read(str(fname_in))

    calc = espresso(pw=cutoffs, 
                                dw=cutoffs*10.,
                                xc='BEEF-vdW',
                                kpts =(6,6,1),
                                nbands=-10,
                                spinpol=spinpol,
                                nosym=True,
                                occupations= 'smearing',
                                smearing = 'fd',
                                output = {'avoidio':False,'removewf':False,'wf_collect':True},
                                #mode='vc-relax',
                                dipole=dipole,
                                convergence=convergence,
                                outdir='cellrelax')
    atoms.set_calculator(calc)
    qn = BFGS(atoms, trajectory='relaxAtoms.traj', logfile='relaxAtoms.log')
    qn.run(fmax=0.05)
    energy = atoms.get_potential_energy()
    write(fname_out[0:-5]+'.png',atoms)
    write(fname_out,atoms)

    return atomToString(atoms)
