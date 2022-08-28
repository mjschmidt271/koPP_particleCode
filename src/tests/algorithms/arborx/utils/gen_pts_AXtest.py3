#!/usr/bin/env python3

# this is a script to generate 1, 2, or 3d points for the arborx unit test

import numpy as np
import yaml
import argparse

# Instantiate the parser and parse them args
parser = argparse.ArgumentParser(description='Generate test points for ArborX test')
parser.add_argument('dim', type=int, help='Number of spatial dimensions (1, 2, 3)',
                    choices=range(1, 4))
parser.add_argument('N', type=int, help='Number of points')
parser.add_argument('L', type=float, help='Side length for hypercubic domain')
parser.add_argument('dist', type=float, help='Search radius')
parser.add_argument('pt_style', type=str,
                    help='equi-spaced (equi, only for 1D) or uniformly scattered (rand)',
                    choices=['equi', 'rand'], default='rand')
parser.add_argument('--fname', type=str, default='test_pts.yaml',
                    help='yaml output file name--e.g., test_pts.yaml')

args = parser.parse_args()

dim = args.dim
N = args.N
L = args.L
dist = args.dist
pt_style = args.pt_style
fname = args.fname

if args.N < 5:
    parser.error('Please choose N > 5')

if args.L <= 0.0:
    parser.error('L must be positive')

if args.dist > L or args.dist < 0.0:
    parser.error('Must have 0 < dist < L')

if pt_style == 'rand':
    pts = np.random.uniform(0, L, [int(N), dim])
elif pt_style == 'equi':
    pts = np.linspace(0, L, N)
else:
    raise ValueError("How did we get here??")

data = dict()
data['N'] = int(N)
data['dim'] = dim
data['L'] = L
data['dist'] = dist
data['pts'] = {}
pts_out = data['pts']
if pt_style == 'rand':
    if dim == 1:
        pts_out['x'] = pts[:, 0].tolist()
    elif dim == 2:
        pts_out['x'] = pts[:, 0].tolist()
        pts_out['y'] = pts[:, 1].tolist()
    elif dim == 3:
        pts_out['x'] = pts[:, 0].tolist()
        pts_out['y'] = pts[:, 1].tolist()
        pts_out['z'] = pts[:, 2].tolist()
elif pt_style == 'equi':
    pts_out['x'] = pts.tolist()
else:
    raise ValueError("How did we get here??")

# open the file and write the yaml
of = open(fname, 'w+')
yaml.safe_dump(data, of)
of.close()
