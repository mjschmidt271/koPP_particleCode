#!/usr/bin/env python3

# this script give a rough, conservative estimate of the number of chunks to use
# in the MT algorithm, based on machine memory and number of particles

import numpy as np
import argparse

# Instantiate the parser and parse them args
parser = argparse.ArgumentParser(description='Estimate proper number of chunks'
                                 'for MT')
parser.add_argument('N_particles', type=float,
                    help='Number of particles to be used in simulation')
mem_group = parser.add_mutually_exclusive_group(required=True)
mem_group.add_argument('--memoryGB', type=int, required=False,
                       help='Machine memory, in GB')
machine_list = ['s104', 's102']
mem_group.add_argument('--machine', type=str, required=False,
                       choices=machine_list,
                       help="Machine name (will not work unless "
                       "you've added it to this script)")

args = parser.parse_args()

Np = int(args.N_particles)
if args.machine is not None:
    machine = str(args.machine)
    if machine == 's104':
        memory = 16
    elif machine == 's102':
        memory = 64
    else:
        raise argparse.ArgumentError(machine, 'We should never get here... but '
                                     'since we are: Invalid machine choice'
                                     "--only supported options are "
                                     "['s102', 's104']")
else:
    memory = int(args.memoryGB)

mem_bytes = (memory - 1.0)*10.0**9
# semi-arbitrarily assume 3 bytes/particle
bytes_perP = 3.0 * 8.0
# conservative calculation, assuming dense matrix, despite the fact that the
# required memory would be some fraction of N^2
Pper_chunk = np.sqrt((mem_bytes / bytes_perP))
n_chunk = int(np.ceil(Np / Pper_chunk))

print(f'For {Np:,} particles, and machine memory of {memory} GB, the '
      f'conservatively estimated choice of chunks is ***~{n_chunk}~***')
